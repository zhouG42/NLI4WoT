# !pip install git+https://github.com/PrithivirajDamodaran/Parrot_Paraphraser.git

from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")

'''
uncomment to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(42)
'''

#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")


with open('/content/input.txt', 'a+') as t2:
  with open('/content/input.txt', 'r+') as t1:

    for i in range(20):
      phrase = t1.readline()
      print(t1.tell())
      print(t2.tell())
      print("---------------------")
      print(phrase)

      para1 = parrot.augment(input_phrase=phrase,
                                  use_gpu=True,
                                  do_diverse=True,
                                  max_return_phrases = 6,
                                  adequacy_threshold = 0.60,
                                  fluency_threshold = 0.60)
      if para1 != None:
          for p in para1:
            p1 = p[0]
            print(p)
            if p1 != phrase:
              #if p[1] > 12:
              t2.write('\n')
              t2.write(p1)
      else:
        pass
