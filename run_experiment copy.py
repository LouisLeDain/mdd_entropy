'''
This scripts runs an expriment to compare the performance of two unsupervised training methods:
    - Conditional Entropy Minimisation, as described in "Robust Unsupervised Adaptation of a Speech Recogniser Using Entropy Minimisation and Speaker Codese
    - Momentum Pseudo-Labeling, as described in 'Improving MD&D with Wav2vec2-based momentum pseudo-labeling for accentedness and intelligibility assessment
    
The experiment uses Wav2Vec2 as the base model and evaluates the performance on the L2-ARCTIC dataset. It follows the following steps:
    1. First fine-tuning with CTC learning
    2. Apply entropy Minimisation to one version of the model
    3. Apply Momentum Pseudo-Labeling to another version of the model
    4. Evaluate both models on the L2-ARCTIC dataset
'''

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torch.nn as nn
from torch.optim import Adam 
from torch.utils.data import DataLoader, Dataset

import os
import librosa

from utils import extract_phoneme_sequence, clean_sequence, greedy_decode, evaluate
from training import train_ctc_model, train_entropy_minimisation, train_pseudo_labeling
from training import ConditionalEntropyLossLog 

# Define the device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the paths

PATH_CTC_TRAINING_AUDIO = "data_bis/wav"
PATH_CTC_TRAINING_TEXTGRID = "data_bis/textgrid"
PATH_UNSUPERVISED_TRAINING_AUDIO = "data_bis/wav"
PATH_TEST = "data_bis"

# Loading the datasets

class AudioDataset(Dataset):
    def __init__(self, audio_dir, textgrid_dir=None):
        """
        Args:
            audio_dir (string): Directory with all the audio files.
            textgrid_dir (string): Directory with all the textgrid files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.audio_dir = audio_dir
        self.textgrid_dir = textgrid_dir
        self.audio_files = os.listdir(audio_dir)
        self.max_audio_length = max([len(librosa.load(os.path.join(audio_dir, f), sr=16000)[0]) for f in self.audio_files if f.endswith('.wav')])

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, _ = librosa.load(audio_path, sr=16000)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        # Pad the waveform to the maximum length
        waveform = torch.nn.functional.pad(waveform, (0, self.max_audio_length - len(waveform)))
    

        # If textgrid_dir is provided, extract the phoneme sequence
        if not self.textgrid_dir:
            return waveform
        
        textgrid_path = os.path.join(self.textgrid_dir, self.audio_files[idx].replace('.wav', '.TextGrid'))
        phoneme_sequence = clean_sequence(extract_phoneme_sequence(textgrid_path))

        element = waveform, phoneme_sequence
        return element
    
def collate_fn(batch):
    # Separate waveforms and phoneme sequences
    waveforms = torch.stack([item[0] for item in batch])
    phoneme_sequences = [item[1] for item in batch]

    return waveforms, phoneme_sequences

def create_dataloader(audio_dir, textgrid_dir, batch_size=16, shuffle=True, collate_fn=None):
    audio_dataset = AudioDataset(audio_dir=audio_dir, textgrid_dir=textgrid_dir)
    if collate_fn is None:
        dataloader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        dataloader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return dataloader

# Create dataloaders

ctc_training_dataloader = create_dataloader(audio_dir=PATH_CTC_TRAINING_AUDIO, textgrid_dir=PATH_CTC_TRAINING_TEXTGRID, batch_size=16, shuffle=True, collate_fn=collate_fn)
unsupervised_training_dataloader = create_dataloader(audio_dir=PATH_UNSUPERVISED_TRAINING_AUDIO, textgrid_dir=None, batch_size=16, shuffle=True)

print(f"Dataloader for CTC training created with {len(ctc_training_dataloader.dataset)} samples.")
print(f"Dataloader for unsupervised training created with {len(unsupervised_training_dataloader.dataset)} samples.")

# Load the Wav2Vec2 models and processors

wav2vec2_baseline_model_base = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base").to(device)
wav2vec2_baseline_processor_base = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

wav2vec2_baseline_model_960 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
wav2vec2_baseline_processor_960 = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Perform CTC training on the baseline model

print("Starting CTC training on the baseline model (base)...")
optimizer_ctc_base = Adam(wav2vec2_baseline_model_base.parameters(), lr=1e-4)

train_ctc_model(wav2vec2_baseline_model_base, 
                wav2vec2_baseline_processor_base, 
                ctc_training_dataloader, 
                optimizer_ctc_base,
                num_epochs=1)

print("Starting CTC training on the baseline model (960h)...")
optimizer_ctc_960 = Adam(wav2vec2_baseline_model_960.parameters(), lr=1e-4)

train_ctc_model(wav2vec2_baseline_model_960, 
                wav2vec2_baseline_processor_960, 
                ctc_training_dataloader, 
                optimizer_ctc_960,
                num_epochs=1)

print("CTC training completed on both baseline models.")

# Save the CTC trained models

wav2vec2_baseline_model_base.save_pretrained("models/wav2vec2_baseline_base_ctc")
wav2vec2_baseline_processor_base.save_pretrained("models/wav2vec2_baseline_base_ctc")

wav2vec2_baseline_model_960.save_pretrained("models/wav2vec2_baseline_960_ctc")
wav2vec2_baseline_processor_960.save_pretrained("models/wav2vec2_baseline_960_ctc")

# Perform entropy minimisation

wav2vec2_entropy_base = Wav2Vec2ForCTC.from_pretrained("models/wav2vec2_baseline_base_ctc").to(device)
wav2vec2_entropy_processor_base = Wav2Vec2Processor.from_pretrained("models/wav2vec2_baseline_base_ctc")

wav2vec2_entropy_960 = Wav2Vec2ForCTC.from_pretrained("models/wav2vec2_baseline_960_ctc").to(device)
wav2vec2_entropy_processor_960 = Wav2Vec2Processor.from_pretrained("models/wav2vec2_baseline_960_ctc")

loss_function_entropy = ConditionalEntropyLossLog(n_best_function=greedy_decode)

print("Starting entropy minimisation on the baseline model (base)...")
optimizer_entropy_base = Adam(wav2vec2_entropy_base.parameters(), lr=1e-4)


train_entropy_minimisation(model=wav2vec2_entropy_base,
                            optimizer=optimizer_entropy_base,
                            loss_function=loss_function_entropy,
                            dataloader=unsupervised_training_dataloader,
                            num_epochs=1)

print("Starting entropy minimisation on the baseline model (960h)...")
optimizer_entropy_960 = Adam(wav2vec2_entropy_960.parameters(), lr=1e-4)

train_entropy_minimisation(model=wav2vec2_entropy_960,
                            optimizer=optimizer_entropy_960,
                            loss_function=loss_function_entropy,
                            dataloader=unsupervised_training_dataloader,
                            num_epochs=1)

print("Entropy minimisation completed on both baseline models.")

# Save the entropy minimised models

wav2vec2_entropy_base.save_pretrained("models/wav2vec2_entropy_base")
wav2vec2_entropy_processor_base.save_pretrained("models/wav2vec2_entropy_base")

wav2vec2_entropy_960.save_pretrained("models/wav2vec2_entropy_960")
wav2vec2_entropy_processor_960.save_pretrained("models/wav2vec2_entropy_960")

# Perform momentum pseudo-labeling

wav2vec2_mpl_base = Wav2Vec2ForCTC.from_pretrained("models/wav2vec2_baseline_base_ctc").to(device)
teacher_wav2vec2_mpl_base = Wav2Vec2ForCTC.from_pretrained("models/wav2vec2_baseline_base_ctc").to(device)
wav2vec2_mpl_processor_base = Wav2Vec2Processor.from_pretrained("models/wav2vec2_baseline_base_ctc")

wav2vec2_mpl_960 = Wav2Vec2ForCTC.from_pretrained("models/wav2vec2_baseline_960_ctc").to(device)
teacher_wav2vec2_mpl_960 = Wav2Vec2ForCTC.from_pretrained("models/wav2vec2_baseline_960_ctc").to(device)
wav2vec2_mpl_processor_960 = Wav2Vec2Processor.from_pretrained("models/wav2vec2_baseline_960_ctc")

loss_function_mpl = nn.CTCLoss()

print("Starting momentum pseudo-labeling on the baseline model (base)...")
optimizer_mpl_base = Adam(wav2vec2_mpl_base.parameters(), lr=1e-4)

train_pseudo_labeling(model = wav2vec2_mpl_base,
                      teacher_model = teacher_wav2vec2_mpl_base, 
                      processor = wav2vec2_mpl_processor_base, 
                      optimizer = optimizer_mpl_base, 
                      loss_function = loss_function_mpl, 
                      dataloader = unsupervised_training_dataloader, 
                      num_epochs=1)

print("Starting momentum pseudo-labeling on the baseline model (960h)...")
optimizer_mpl_960 = Adam(wav2vec2_mpl_960.parameters(), lr=1e-4)

train_pseudo_labeling(model = wav2vec2_mpl_960,
                        teacher_model = teacher_wav2vec2_mpl_960,
                        processor = wav2vec2_mpl_processor_960,
                        optimizer = optimizer_mpl_960,
                        loss_function = loss_function_mpl,
                        dataloader = unsupervised_training_dataloader,
                        num_epochs=1)

print("Momentum pseudo-labeling completed on both baseline models.")

# Save the momentum pseudo-labeled models

wav2vec2_mpl_base.save_pretrained("models/wav2vec2_mpl_base")
wav2vec2_mpl_processor_base.save_pretrained("models/wav2vec2_mpl_base")

wav2vec2_mpl_960.save_pretrained("models/wav2vec2_mpl_960")
wav2vec2_mpl_processor_960.save_pretrained("models/wav2vec2_mpl_960")

# Evaluate the models

print("Evaluating the entropy minimised model (base)...")
evaluate(model=wav2vec2_entropy_base,
        processor=wav2vec2_entropy_processor_base,
        test_dataset=PATH_TEST)

print("Evaluating the entropy minimised model (960h)...")
evaluate(model=wav2vec2_entropy_960,
        processor=wav2vec2_entropy_processor_960,
        test_dataset=PATH_TEST)

print("Evaluating the momentum pseudo-labeled model (base)...")
evaluate(model=wav2vec2_mpl_base,
        processor=wav2vec2_mpl_processor_base,
        test_dataset=PATH_TEST)

print("Evaluating the momentum pseudo-labeled model (960h)...")
evaluate(model=wav2vec2_mpl_960,
        processor=wav2vec2_mpl_processor_960,
        test_dataset=PATH_TEST)

