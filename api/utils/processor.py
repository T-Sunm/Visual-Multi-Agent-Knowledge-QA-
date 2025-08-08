import torch
from torchvision import transforms
from PIL import Image
from typing import List
from underthesea import word_tokenize
class Processor:
    """
    Processor for handling image and text inputs for the ViVQA-X model.
    """
    def __init__(self, word2idx, max_question_length=20):
        """
        Initializes the processor.

        Args:
            word2idx (dict): Vocabulary mapping words to indices.
            max_question_length (int): Maximum length for tokenized questions.
        """
        self.word2idx = word2idx
        self.max_question_length = max_question_length
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def tokenize(self, text: str) -> List[int]:
        """Tokenizes text and maps words to indices."""
        tokens = word_tokenize(text.lower())
        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]

    def pad_sequence(self, sequence: List[int], max_length: int) -> List[int]:
        """Pads or truncates a sequence to a fixed length."""
        if len(sequence) > max_length:
            return sequence[:max_length]
        return sequence + [self.word2idx['<PAD>']] * (max_length - len(sequence))

    def __call__(self, image: Image.Image, question: str):
        """
        Processes an image and a question.

        Args:
            image (PIL.Image.Image): The input image.
            question (str): The input question.

        Returns:
            dict: A dictionary containing processed image and question tensors.
        """
        # Process image
        processed_image = self.transform(image)

        # Process question
        question_tokens = self.tokenize(question)
        padded_question = self.pad_sequence(question_tokens, self.max_question_length)
        processed_question = torch.LongTensor(padded_question)

        return {
            "image": processed_image.unsqueeze(0), # Add batch dimension
            "question": processed_question.unsqueeze(0) # Add batch dimension
        } 