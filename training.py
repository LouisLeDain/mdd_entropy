import torch
from utils import clean_sequence
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self, n_best_function, N=5):
        super(ConditionalEntropyLoss, self).__init__()
        self.n_best_function = n_best_function
        self.N = N
        
    def forward(self, inputs, epsilon=1e-10): # inputs is a frame_wise probabilities of shape (num_frames, num_classes)
        n_best = self.n_best_function(inputs, self.N) # shape list of dictionaries with 'sequence' and 'probability' keys
        z_inputs = torch.tensor(0.0, dtype = torch.float64, requires_grad=True)
        entropy = torch.tensor(0.0, dtype = torch.float64, requires_grad=True)

        for i in range(len(n_best)):
            prob = torch.tensor(n_best[i]['probability'], requires_grad= True)  # Log Probability of the sequence
            # print(f"Probability of sequence {i}: {prob.item()}")
            z_inputs = z_inputs + torch.exp(prob)  # Sum of exponentials of log probabilities
            entropy = entropy + prob * torch.exp(prob)  # Weighted sum of log probabilities

        # print(f"Z Inputs: {z_inputs}")        
        entropy = -entropy / (z_inputs + epsilon)
        # print(f"Entropy: {entropy}")
        return entropy
    
class ConditionalEntropyLossLog(torch.nn.Module):
    def __init__(self, n_best_function, N=5):
        super(ConditionalEntropyLossLog, self).__init__()
        self.n_best_function = n_best_function
        self.N = N
        
    def forward(self, inputs): # inputs is a frame_wise probabilities of shape (num_frames, num_classes)
        n_best = self.n_best_function(inputs, self.N) # shape list of dictionaries with 'sequence' and 'probability' keys
        log_entity = torch.tensor(0.0, dtype = torch.float64, requires_grad=True)
        entropy = torch.tensor(0.0, dtype = torch.float64, requires_grad=True)

        probs = torch.tensor([n_best[i]['probability'] for i in range(len(n_best))], requires_grad=True) # list of log-probabilities
        prob_max = torch.max(probs)

        for i in range(len(probs)):
            log_entity = log_entity + torch.exp(probs[i]-prob_max)
        
        log_entity = torch.log(log_entity)

        for i in range(len(probs)):
            entropy = entropy + probs[i] * torch.exp(probs[i] - prob_max - log_entity)

        entropy = - entropy
        # print(f"Entropy: {entropy}")
        return entropy

def train_ctc_model(model, processor, dataloader, optimizer, num_epochs=10):
    loss_function = nn.CTCLoss().to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            waveforms, phoneme_sequences = batch
            batch_loss = torch.tensor(0.0, requires_grad=True)
            optimizer.zero_grad()
            for i in range(len(waveforms)):
                input = waveforms[i].unsqueeze(0).to(device)  # Add batch dimension and move to device
                outputs = model(input).logits # shape (batch_size, sequence_length, num_classes)
                log_probs = outputs.log_softmax(dim=-1).permute(1, 0, 2)  # shape (sequence_length, batch_size, num_classes)

                target = phoneme_sequences[i]  # Get the target phoneme sequence
                target = processor.tokenizer(target, return_tensors="pt", padding=True).input_ids.to(device)  # Convert to tensor and move to device
                
                input_lengths = torch.tensor([log_probs.shape[0]]).to(device)  # Length of each input in the batch
                target_lengths = torch.tensor([len(t) for t in target], dtype=torch.int32).to(device)  # Length of each target in the batch
                
                loss = loss_function(log_probs, target, input_lengths, target_lengths)  # Compute the CTC loss 
                batch_loss = batch_loss + loss.item()
            batch_loss = batch_loss / len(batch)
            print(f"Batch loss: {batch_loss}")
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

def train_entropy_minimisation(model, optimizer, loss_function, dataloader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            batch_loss = torch.tensor(0.0, dtype = torch.float64, requires_grad=True)
            optimizer.zero_grad()
            for _, input in enumerate(batch):
                input = input.unsqueeze(0).to(device)
                outputs = model(input).logits
                probs = outputs.softmax(dim=-1)
                loss = loss_function(probs.squeeze(0))  # Squeeze to remove batch dimension
                # print(f"Loss for current input: {loss}")
                batch_loss = batch_loss + loss.item()

            batch_loss = batch_loss / len(batch)
            print(f"Batch loss: {batch_loss}")
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

def train_pseudo_labeling(model, teacher_model, processor, optimizer, loss_function, dataloader, num_epochs=10, alpha=0.9):
    for epoch in range(num_epochs):
        model.train()
        teacher_model.eval()  # Ensure the teacher model is in evaluation mode
        total_loss = 0.0
        for batch in dataloader:
            batch_loss = torch.tensor(0.0, requires_grad=True)
            optimizer.zero_grad()
            for _, input in enumerate(batch):
                input = input.unsqueeze(0).to(device)
                outputs = model(input).logits # shape (batch_size, sequence_length, num_classes)
                log_probs = outputs.log_softmax(dim=-1).permute(1, 0, 2)  # shape (sequence_length, batch_size, num_classes)

                target = teacher_model(input).logits  # Get the teacher model's logits
                target_ids = torch.argmax(target, dim=-1)  # Get the predicted IDs from the teacher model
                target_decoded = clean_sequence(processor.batch_decode(target_ids)[0])  # Decode the target IDs
                target = processor.tokenizer(target_decoded, return_tensors="pt", padding=True).input_ids.to(device)  # Convert to tensor and move to device
                
                input_lengths = torch.tensor([log_probs.shape[0]]).to(device)  # Length of each input in the batch
                target_lengths = torch.tensor([len(t) for t in target], dtype=torch.int32).to(device)  # Length of each target in the batch
                
                loss = loss_function(log_probs, target, input_lengths, target_lengths)  # Compute the CTC loss 
                batch_loss = batch_loss + loss.item()
            batch_loss = batch_loss / len(batch)
            print(f"Batch loss: {batch_loss}")
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Update the teacher model with the student model weights
        for teacher_param, student_param in zip(teacher_model.parameters(), model.parameters()):
            teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
