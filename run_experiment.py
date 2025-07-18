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

L2ARCTIC_DIR = "/app/data"

test_folders = ['HQTV', 'ASI', 'EBVS', 'ABA']

list_of_test_files = {name : {'wav' : [os.path.join(name, 'wav', file.replace('.TextGrid', '.wav')) for file in sorted(os.listdir(os.path.join(L2ARCTIC_DIR, name, "annotation")))],
                              'textgrid' : [os.path.join(name, 'annotation', file) for file in sorted(os.listdir(os.path.join(L2ARCTIC_DIR, name, "annotation")))]}  for name in test_folders}

train_folders = ['HKK', 'MBMPS', 'PNV', 'SVBI', 'TNI', 'YDCK', 'ERMS', 'NCC', 'RRBI', 'THV', 'TXHC', 'YKWK', 'BWC', 'HJK', 'LXC', 'NJS', 'SKA', 'TLV', 'YBAA', 'ZHAA']
list_of_train_ctc_files = {name : {'wav' : [os.path.join(name, 'wav', file.replace('.TextGrid', '.wav')) for file in sorted(os.listdir(os.path.join(L2ARCTIC_DIR, name, "annotation")))],
                              'textgrid' : [os.path.join(name, 'annotation', file) for file in sorted(os.listdir(os.path.join(L2ARCTIC_DIR, name, "annotation")))]}  for name in train_folders}

list_of_unsupervised_files = {name : {'wav' : [os.path.join(name, 'wav', file) for file in sorted(os.listdir(os.path.join(L2ARCTIC_DIR, name,'wav'))) if os.path.join(name, 'annotation', file.replace('.wav', '.TextGrid')) not in list_of_train_ctc_files[name]['textgrid']]} for name in train_folders}


# Loading the datasets

class AudioDataset(Dataset):
    def __init__(self, data_dir, data, textgrid = False):
        """
        Args:
            audio_dir (string): Directory with all the audio files.
            textgrid_dir (string): Directory with all the textgrid files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.textgrid = textgrid
        if textgrid:
            self.list_of_files = {'wav': [], 'textgrid': []}
        else:
            self.list_of_files = {'wav': []}

        for folder in data.keys():
            self.list_of_files['wav'].extend(data[folder]['wav'])
            if textgrid:
                self.list_of_files['textgrid'].extend(data[folder]['textgrid'])

        self.max_length = max([len(librosa.load(os.path.join(self.data_dir, file), sr=16000, mono=True)[0]) for file in self.list_of_files['wav']])

    def __len__(self):
        return len(self.list_of_files['wav'])

    def __getitem__(self, idx):
        audio_path = os.path.join(self.data_dir, self.list_of_files['wav'][idx])
        if self.textgrid:
            textgrid_path = os.path.join(self.data_dir, self.list_of_files['textgrid'][idx])
            phoneme_sequence = clean_sequence(extract_phoneme_sequence(textgrid_path))
            waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
            waveform = torch.from_numpy(waveform)
            # Pad the waveform to the maximum length
            waveform = torch.nn.functional.pad(waveform, (0, self.max_length - len(waveform)))
            return waveform, phoneme_sequence
        else:
            waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
            waveform = torch.from_numpy(waveform)
            # Pad the waveform to the maximum length
            waveform = torch.nn.functional.pad(waveform, (0, self.max_length - len(waveform)))
            return waveform
    
def collate_fn(batch):
    # Separate waveforms and phoneme sequences
    waveforms = torch.stack([item[0] for item in batch])
    phoneme_sequences = [item[1] for item in batch]

    return waveforms, phoneme_sequences

def create_dataloader(data_dir, data, batch_size=16, shuffle=True, collate_fn=None):
    audio_dataset = AudioDataset(data_dir = data_dir,data = data, textgrid=(collate_fn is not None))
    if collate_fn is None:
        dataloader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        dataloader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return dataloader

# Create dataloaders

ctc_training_dataloader = create_dataloader(data_dir=L2ARCTIC_DIR,
                                            data=list_of_train_ctc_files,
                                            batch_size=16,
                                            shuffle=True,
                                            collate_fn=collate_fn)

unsupervised_training_dataloader = create_dataloader(data_dir=L2ARCTIC_DIR,
                                                    data=list_of_unsupervised_files,
                                                    batch_size=16,
                                                    shuffle=True)

test_dataloader = create_dataloader(data_dir=L2ARCTIC_DIR,
                                    data=list_of_test_files,
                                    batch_size=16,
                                    shuffle=False,
                                    collate_fn=collate_fn)

print(f"Dataloader for CTC training created with {len(ctc_training_dataloader.dataset)} samples.")
print(f"Dataloader for unsupervised training created with {len(unsupervised_training_dataloader.dataset)} samples.")
print(f"Dataloader for testing created with {len(test_dataloader.dataset)} samples.")

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
        test_dataloader=test_dataloader)

print("Evaluating the entropy minimised model (960h)...")
evaluate(model=wav2vec2_entropy_960,
        processor=wav2vec2_entropy_processor_960,
        test_dataloader=test_dataloader)

print("Evaluating the momentum pseudo-labeled model (base)...")
evaluate(model=wav2vec2_mpl_base,
        processor=wav2vec2_mpl_processor_base,
        test_dataloader=test_dataloader)

print("Evaluating the momentum pseudo-labeled model (960h)...")
evaluate(model=wav2vec2_mpl_960,
        processor=wav2vec2_mpl_processor_960,
        test_dataloader=test_dataloader)

