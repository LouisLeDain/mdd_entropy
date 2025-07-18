import torch
from textgrid import TextGrid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_phoneme_sequence(textgrid_file_path):
    try: 
        # Read the TextGrid file
        textgrid = TextGrid.fromFile(textgrid_file_path)

        # Assuming the phoneme tier is named 'phones'
        phoneme_tier = textgrid.getFirst("phones")

        if phoneme_tier is None:
            raise ValueError("No phoneme tier found in the TextGrid file.")

        # Extract phoneme sequence
        phoneme_sequence = [interval.mark for interval in phoneme_tier.intervals if interval.mark]
        return ' '.join(phoneme_sequence)
    except Exception as e:
        print(f"Error extracting phoneme sequence from {textgrid_file_path}: {e}")
        return None

def clean_sequence(sequence):
    """
    Cleans the phoneme sequence by removing numbers, 'sil' and 'sp' elements.
    """
    cleaned_sequence = []
    for element in sequence.split(' '):
        if any(char.isdigit() for char in element) :
            element = element[:-1]  # Remove the last character if it's a digit
            cleaned_sequence.append(element)
        elif any(char == ',' for char in element) :
            chars = element.split(',')[1]
            if chars != 'sil':
                cleaned_sequence.append(chars)
        elif element == 'A':
            cleaned_sequence.append('AA')
        elif element == 'H':
            cleaned_sequence.append('HH')
        elif element not in ['sil', 'sp'] and element != '':
            cleaned_sequence.append(element)
    return ' '.join(cleaned_sequence)

def greedy_decode(frame_wise_probs, N=5):
    """
    Perform greedy decoding to find the N most likely sequences from frame-wise probabilities.

    :param frame_wise_probs: A torch tensor of shape (sequence_length, num_classes) containing frame-wise probabilities.
    :param N: The number of most likely sequences to return.
    :return: A list of the N most likely sequences and their probabilities.
    """
    # Check if N is valid
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    # Convert probabilities to log probabilities for numerical stability
    log_probs = torch.log(frame_wise_probs)

    # Initialize the sequences and their probabilities
    sequences = [{
        'sequence': [],
        'probability': torch.tensor(0.0, dtype = torch.float64, requires_grad=True)
    }]

    # Iterate over each frame
    for t in range(log_probs.shape[0]):
        current_frame_probs = log_probs[t]
        new_sequences = []

        # Expand each sequence with the most probable label at the current frame
        for seq in sequences:
            # Find the most probable label at the current frame
            best_labels = torch.topk(current_frame_probs, N).indices
            best_label_probs = current_frame_probs[best_labels]
            
            for i, best_label in enumerate(best_labels):
                best_label_prob = best_label_probs[i]

                # Create a new sequence by appending the best label
                new_sequence = seq['sequence'] + [best_label.item()]
                new_probability = seq['probability'] + best_label_prob

                new_sequences.append({
                    'sequence': new_sequence,
                    'probability': new_probability
                })

        # Sort sequences by their probabilities and keep the top N sequences
        new_sequences.sort(key=lambda x: x['probability'], reverse=True)
        sequences = new_sequences[:N]

    # Return the N most likely sequences and their probabilities
    return sequences

def calculate_per(reference, hypothesis): # inputs are lists of phonemes obtained with .split(' ')

    d = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]

    # Initialize the first row and column of the matrix
    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(hypothesis) + 1):
        d[0][j] = j

    # Fill in the matrix
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # Deletion
                d[i][j - 1] + 1,      # Insertion
                d[i - 1][j - 1] + cost  # Substitution
            )

    # The last element of the matrix contains the Levenshtein distance
    distance = d[len(reference)][len(hypothesis)]

    # Calculate the Phoneme Error Rate
    per = distance / len(reference)
    return per

def calculate_metrics(reference, hypothesis):
    # Convert sequences to lists of phonemes
    ref_phonemes = reference.split()
    hyp_phonemes = hypothesis.split()

    # Create a matrix to store the costs of deletions, insertions, and substitutions
    d = [[0] * (len(hyp_phonemes) + 1) for _ in range(len(ref_phonemes) + 1)]

    # Initialize the first row and column of the matrix
    for i in range(len(ref_phonemes) + 1):
        d[i][0] = i
    for j in range(len(hyp_phonemes) + 1):
        d[0][j] = j

    # Fill in the matrix
    for i in range(1, len(ref_phonemes) + 1):
        for j in range(1, len(hyp_phonemes) + 1):
            if ref_phonemes[i - 1] == hyp_phonemes[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # Deletion
                d[i][j - 1] + 1,      # Insertion
                d[i - 1][j - 1] + cost  # Substitution
            )

    # Backtrack to find the alignment
    i, j = len(ref_phonemes), len(hyp_phonemes)
    correct_predictions = 0

    while i > 0 and j > 0:
        if ref_phonemes[i - 1] == hyp_phonemes[j - 1]:
            correct_predictions += 1
            i -= 1
            j -= 1
        else:
            min_val = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
            if d[i - 1][j] == min_val:
                i -= 1
            elif d[i][j - 1] == min_val:
                j -= 1
            else:
                i -= 1
                j -= 1

    # Calculate precision, recall
    precision = correct_predictions / len(hyp_phonemes) if hyp_phonemes else 0
    recall = correct_predictions / len(ref_phonemes) if ref_phonemes else 0

    return precision, recall

def evaluate(model, processor, test_dataloader):
    """
    Evaluate the model through the proper metrics : 
    - Phoneme Error Rate (PER)
    - F1 Score
    """

    with torch.no_grad():
        total_per = 0
        total_precision = 0
        total_recall = 0
        
        for idx in range(len(test_dataloader.dataset)):
            wav_file, target_text = test_dataloader.dataset[idx]
            if wav_file is not None:
                # Load the audio file
                input_values = processor(wav_file, sampling_rate=16000, return_tensors="pt").input_values.to(device)

                # Get model predictions
                outputs = model(input_values).logits
                predicted_ids = torch.argmax(outputs, dim=-1)
                transcription = clean_sequence(processor.batch_decode(predicted_ids)[0])

                per = calculate_per(target_text.split(' '), transcription.split(' '))
                precision, recall = calculate_metrics(target_text, transcription)

                total_per += per
                total_precision += precision
                total_recall += recall

        # Calculate average metrics
        num_files = len(test_dataloader.dataset)
        avg_per = total_per / num_files if num_files > 0 else 0
        avg_precision = total_precision / num_files if num_files > 0 else 0
        avg_recall = total_recall / num_files if num_files > 0 else 0
        f1_score = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        print(f"Average PER: {avg_per:.4f}, Average Precision: {avg_precision:.4f}, Average Recall: {avg_recall:.4f}, F1 Score: {f1_score:.4f}")
