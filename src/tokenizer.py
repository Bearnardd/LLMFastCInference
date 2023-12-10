from sentencepiece import SentencePieceProcessor
import os
from typing import List
import struct
import argparse


class Tokenizer:
    def __init__(self, model_path):
        model_path = model_path if model_path is not None else "tokenizer.model"
        assert os.path.isfile(model_path), model_path 
        self.model_path = model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.sp_model.bos_id()] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def export(self) -> None:
        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):
            # decode the token and light postprocessing
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace
            b = t.encode('utf-8') # bytes of this token, utf-8 encoded
            tokens.append(b)
            scores.append(s)
        # record the max token length
        max_token_length = max(len(t) for t in tokens)
        # write to a binary file
        # the tokenizer.bin file is the same as .model file, but .bin
        tokenizer_bin = self.model_path.replace('.model', '.bin')
        with open(tokenizer_bin, 'wb') as f:
            f.write(struct.pack("I", max_token_length))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer-model", type=str, help="optional path to custom tokenizer ")
    args = parser.parse_args()

    t = Tokenizer(args.tokenizer_model)
    t.export()
# if __name__ == "__main__":
#     s = """The code snippet you've provided demonstrates a common and effective way to download files using Python's
#             requests library while displaying a progress bar using tqdm. This is a clear and concise approach to download
#             files with a progress indicator, offering readability and functionality. Here's a brief explanation of what
#             this code does: It uses the requests library (resp variable) to make an HTTP request to download a file.
#             The resp.iter_content(chunk_size=chunk_size) method iterates over the content of the response in chunks.
#             The downloaded content is written to a file (fname) in binary mode ("wb"). The tqdm library is used to
#             create a progress bar (bar) based on the total file size (total), and it updates as chunks of data are
#             received and written to the file. This code provides a clear and efficient way to download files, allowing
#             for a visual progress indicator using tqdm. The use of context managers (with open(...) as file) ensures that
#             the file is properly opened, written, and closed, preventing resource leaks. If you're looking for an 
#             alternative method or improvement, it largely depends on specific requirements or preferences. However,
#             the approach you've shown is commonly used and provides a clear and readable way to download files with
#             a progress bar. The code snippet you've provided demonstrates a common and effective way to download files using Python's
#             requests library while displaying a progress bar using tqdm. This is a clear and concise approach to download
#             files with a progress indicator, offering readability and functionality. Here's a brief explanation of what
#             this code does: It uses the requests library (resp variable) to make an HTTP request to download a file.
#             The resp.iter_content(chunk_size=chunk_size) method iterates over the content of the response in chunks.
#             The downloaded content is written to a file (fname) in binary mode ("wb"). The tqdm library is used to
#             create a progress bar (bar) based on the total file size (total), and it updates as chunks of data are
#             received a    def decode(self, t: List[int]) -> str:
#         return self.sp_model.decode(t)
# nd written to the file. This code provides a clear and efficient way to download files, allowing
#             for a visual progress indicator using tqdm. The use of context managers (with open(...) as file) ensures that
#             the file is properly opened, written, and closed, preventing resource leaks. If you're looking for an 
#             alternative method or improvement, it largely depends on specific requirements or preferences. However,
#             the approach you've shown is commonly used and provides a clear and readable way to download files with
#             a progress bar. he code snippet you've provided demonstrates a common and effective way to download files using Python's
#             requests library while displaying a progress bar using tqdm. This is a clear and concise approach to download
#             files with a progress indicator, offering readability and functionality. Here's a brief explanation of what
#             this code does: It uses the requests library (resp variable) to make an HTTP request to download a file.
#             The resp.iter_content(chunk_size=chunk_size) method iterates over the content of the response in chunks.
#             The downloaded content is written to a file (fname) in binary mode ("wb"). The tqdm library is used to
#             create a progress bar (bar) based on the total file size (total), and it updates as chunks of data are
#             received and written to the file. This code provides a clear and efficient way to download files, allowing
#             for a visual progress indicator using tqdm. The use of context managers (with open(...) as file) ensures that
#             the file is properly opened, written, and closed, preventing resource leaks. If you're looking for an 
#             alternative method or improvement, it largely depends on specific requirements or preferences. However,
#             the approach you've shown is commonly used and provides a clear and readable way to download files with
#             a progress bar. he code snippet you've provided demonstrates a common and effective way to download files using Python's
#             requests library while displaying a progress bar using tqdm. This is a clear and concise approach to download
#             files with a progress indicator, offering readability and functionality. Here's a brief explanation of what
#             this code does: It uses the requests library (resp variable) to make an HTTP request to download a file.
#             The resp.iter_content(chunk_size=chunk_size) method iterates over the content of the response in chunks.
#             The downloaded content is written to a file (fname) in binary mode ("wb"). The tqdm library is used to
#             create a progress bar (bar) based on the total file size (total), and it updates as chunks of data are
#             received and written to the file. This code provides a clear and efficient way to download files, allowing
#             for a visual progress indicator using tqdm. The use of context managers (with open(...) as file) ensures that
#             the file is properly opened, written, and closed, preventing resource leaks. If you're looking for an 
#             alternative method or improvement, it largely depends on specific requirements or preferences. However,
#             the approach you've shown is commonly used and provides a clear and readable way to download files with
#             a progress bar. he code snippet you've provided demonstrates a common and effective way to download files using Python's
#             requests library while displaying a progress bar using tqdm. This is a clear and concise approach to download
#             files with a progress indicator, offering readability and functionality. Here's a brief explanation of what
#             this code does: It uses the requests library (resp variable) to make an HTTP request to download a file.
#             The resp.iter_content(chunk_size=chunk_size) method iterates over the content of the response in chunks.
#             The downloaded content is written to a file (fname) in binary mode ("wb"). The tqdm library is used to
#             create a progress bar (bar) based on the total file size (total), and it updates as chunks of data are
#             received and written to the file. This code provides a clear and efficient way to download files, allowing
#             for a visual progress indicator using tqdm. The use of context managers (with open(...) as file) ensures that
#             the file is properly opened, written, and closed, preventing resource leaks. If you're looking for an 
#             alternative method or improvement, it largely depends on specific requirements or preferences. However,
#             the approach you've shown isif __name__ == "__main__":
#     s = """The code snippet you've provided demonstrates a common and effective way to download files using Python's
#             requests library while displaying a progress bar using tqdm. This is a clear and concise approach to download
#             files with a progress indicator, offering readability and functionality. Here's a brief explanation of what
#             this code does: It uses the requests library (resp variable) to make an HTTP request to download a file.
#             The resp.iter_content(chunk_size=chunk_size) method iterates over the content of the response in chunks.
#             The downloaded content is written to a file (fname) in binary mode ("wb"). The tqdm library is used to
#             create a progress bar (bar) based on the total file size (total), and it updates as chunks of data are
#             received and written to the file. This code provides a clear and efficient way to download files, allowing
#             for a visual progress indicator using tqdm. The use of context managers (with open(...) as file) ensures that
#             the file is properly opened, written, and closed, preventing resource leaks. If you're looking for an 
#             alternative method or improvement, it largely depends on specific requirements or preferences. However,
#             the approach you've shown is commonly used and provides a clear and readable way to download files with
#             a progress bar. The code snippet you've provided demonstrates a common and effective way to download files using Python's
#             requests library while displaying a progress bar using tqdm. This is a clear and concise approach to download
#             files with a progress indicator, offering readability and functionality. Here's a brief explanation of what
#             this code does: It uses the requests library (resp variable) to make an HTTP request to download a file.
#             The resp.iter_content(chunk_size=chunk_size) method iterates over the content of the response in chunks.
#             The downloaded content is written to a file (fname) in binary mode ("wb"). The tqdm library is used to
#             create a progress bar (bar) based on the total file size (total), and it updates as chunks of data are
#             received a    def decode(self, t: List[int]) -> str:
#         return self.sp_model.decode(t)
# nd written to the file. This code provides a clear and efficient way to download files, allowing
#             for a visual progress indicator using tqdm. The use of context managers (with open(...) as file) ensures that
#             the file is properly opened, written, and closed, preventing resource leaks. If you're looking for an 
#             alternative method or improvement, it largely depends on specific requirements or preferences. However,
#             the approach you've shown is commonly used and provides a clear and readable way to download files with
#             a progress bar. he code snippet you've provided demonstrates a common and effective way to download files using Python's
#             requests library while displaying a progress bar using tqdm. This is a clear and concise approach to download
#             files with a progress indicator, offering readability and functionality. Here's a brief explanation of what
#             this code does: It uses the requests library (resp variable) to make an HTTP request to download a file.
#             The resp.iter_content(chunk_size=chunk_size) method iterates over the content of the response in chunks.
#             The downloaded content is written to a file (fname) in binary mode ("wb"). The tqdm library is used to
#             create a progress bar (bar) based on the total file size (total), and it updates as chunks of data are
#             received and written to the file. This code provides a clear and efficient way to download files, allowing
#             for a visual progress indicator using tqdm. The use of context managers (with open(...) as file) ensures that
#             the file is properly opened, written, and closed, preventing resource leaks. If you're looking for an 
#             alternative method or improvement, it largely depends on specific requirements or preferences. However,
#             the approach you've shown is commonly used and provides a clear and readable way to download files with
#             a progress bar. he code snippet you've provided demonstrates a common and effective way to download files using Python's
#             requests library while displaying a progress bar using tqdm. This is a clear and concise approach to download
#             files with a progress indicator, offering readability and functionality. Here's a brief explanation of what
#             this code does: It uses the requests library (resp variable) to make an HTTP request to download a file.
#             The resp.iter_content(chunk_size=chunk_size) method iterates over the content of the response in chunks.
#             The downloaded content is written to a file (fname) in binary mode ("wb"). The tqdm library is used to
#             create a progress bar (bar) based on the total file size (total), and it updates as chunks of data are
#             received and written to the file. This code provides a clear and efficient way to download files, allowing
#             for a visual progress indicator using tqdm. The use of context managers (with open(...) as file) ensures that
#             the file is properly opened, written, and closed, preventing resource leaks. If you're looking for an 
#             alternative method or improvement, it largely depends on specific requirements or preferences. However,
#             the approach you've shown is commonly used and provides a clear and readable way to download files with
#             a progress bar. he code snippet you've provided demonstrates a common and effective way to download files using Python's
#             requests library while displaying a progress bar using tqdm. This is a clear and concise approach to download
#             files with a progress indicator, offering readability and functionality. Here's a brief explanation of what
#             this code does: It uses the requests library (resp variable) to make an HTTP request to download a file.
#             The resp.iter_content(chunk_size=chunk_size) method iterates over the content of the response in chunks.
#             The downloaded content is written to a file (fname) in binary mode ("wb"). The tqdm library is used to
#             create a progress bar (bar) based on the total file size (total), and it updates as chunks of data are
#             received and written to the file. This code provides a clear and efficient way to download files, allowing
#             for a visual progress indicator using tqdm. The use of context managers (with open(...) as file) ensures that
#             the file is properly opened, written, and closed, preventing resource leaks. If you're looking for an 
#             alternative method or improvement, it largely depends on specific requirements or preferences. However,
#             the approach you've shown is commonly used and provides a clear and readable way to download files with
#             a progress bar.jkljlkj""" commonly used and provides a clear and readable way to download files with
#             a progress bar.jkljlkj"""

    # s = "hello, what is going on? What is your name?"
    
    # from transformers import AutoTokenizer
    # import time
    # tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    # t = Tokenizer()
    # start_time = time.time()
    # enc = t.encode(s, bos=True, eos=False)
    # end_time = time.time()
    # duration = end_time - start_time
    # print(f"The function took {duration} seconds to complete.")
    # start_time = time.time()
    # dec = t.decode(enc)
    # end_time = time.time()
    # duration = end_time - start_time
    # print(f"The function took {duration} seconds to complete.")
    # print(dec)
    # start_time = time.time()
    # enc = tok.encode(s)
    # end_time = time.time()
    # duration = end_time - start_time
    # print(f"The function took {duration} seconds to complete.")
    # start_time = time.time()
    # dec = tok.decode(enc)
    # print(dec)
    # end_time = time.time()
    # duration = end_time - start_time
    # print(f"The function took {duration} seconds to complete.")



    # print(t.encode("Hello, what is going on?", bos=True, eos=False))

