# Files and folders structure
* TODO: add in and output of each file
* main.py: the main and interactive script to run train, test, and inference.
  * input: name of the corpus, name of the model, and the task to run (train/test/inference) 
  * output: depends on -t, either a trained model weights or a test score or a converted code  

* train.py: script to train a model given a corpus data
  * input: training corpus in csv file(pairs of natural language and code)
  * output: trained model weights and context mappings
  * 
* test.py: script to evaluate performance (precision) after training
    * input: testing corpus in csv file
    * output: a score indicating trained precision metrics (can print or save to a log file for each command decoding result)
  
* inference.py: script to inference new unseen input, convert nl to code
  * input: a single natural language command
  * output: converted blast code
  

* models.py: define model structure, sequence to sequence models implementation in tensorflow keras (basicSeq2seq, attentionSeq2seq), can be extended to fine-tuned BART or other LLMs.
  * input: no
  * output: no
  * 
* utils.py: helper functions for example text preprocessing
  * input: no
  * output: no
  * 
* /trained_models: saved/cached trained model weights, for later tfjs convertion
* /tfjs_model: converted tfjs models for blast execution


# Usage
## 1. python part
* User can decide whether to train a new model, or evaluate a existing model accuracy, or use an existing model for inference using the ```main.py``` script as below. The command line prompt will then ask the user which device corpus to use, and which model to use.
  ```
  python main.py -t [train | test | inference]
  ```
* Before you have any trained models in your local directory, you should run the training process first with the following script:

  ```
  python main.py -t train
  ```
* Once the model finished training, the trained model (including vocabulary and context information) will be saved in the ```./trained_model``` folder under the corresponding device name folder. For example, if you want to train a model for philips hue, all the saved files after training should be saved in ```./trained_model/philipshue```


* Now you can inference with your trained model with the script below. Follow the prompt for more information.

  ```
  python main.py -t inference
  ```

## 2. tfjs part

* To use the trained model for blast engine, we need to first convert the saved model `.h5` format to `json` with the following command. (Note: the folder path might need to updated). 

  ```
  bash tfjs_converter.sh
  ``` 
* Save the generated model in the `/tfjs/encoder` and `/tfjs/decoder` to a new directory, where the javascript codes for blast execution are saved.
* 



