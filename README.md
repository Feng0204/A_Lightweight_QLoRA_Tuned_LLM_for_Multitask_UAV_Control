# File Description

- `dada` folder contains training and validation data  
- `results` folder contains the checkpoint with the lowest loss saved during training and the final model  
- `loop_session.py` is the multi-turn dialogue code file  
- `QLORA.py` is the model training file  
- `single.py` is the single-turn dialogue code  
- `test_base_model.py` is used to test the downloaded pretrained model  
- `tokena.py` is for tokenization  

# Usage Instructions

## 1. Virtual Environment  
`requirements.txt` lists some of the required libraries (the PyTorch version is for Linux)  

## 2. Download Model  
Model download link: [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct)  
You can automatically load and test the model using the [code](test_base_model.py).  
**Note:** Change the path in line 4 to your local path.  

## 3. Model Training  
`QLORA.py` contains the model training code. Modify necessary parts as needed.  

## 4. Running the Code  
The multi-turn dialogue code file is `loop_session.py`. Run it directly without modification.  
(Due to relative paths, you need to run this code from the `coderV2` directory.)  

The workflow: after entering a request, the model generates code.  
Then input `y/n` to decide whether to use the generated code.  
- Enter `n` to continue entering instructions to modify the code.  
- Enter `y` to accept the code and clear the history.  
