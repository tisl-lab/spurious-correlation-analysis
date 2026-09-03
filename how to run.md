1- fine-tune clip to make sure spurrious correlation exists
it generates and saves, clip model and manifest files
run_waterbirds_msae.py
2- start with precompute_activations, and run it for the waterbirds dataset on the clip model. to adjust to our fine_tuned implementation we need to have some updates on the code to get fine_tuned model's embeddings(update implemented for ft_model loading -June 09), and also the predefined dataset used for fine tunning. in the current setting everywhere the zero shot clip is being used and also full training set of the waterbirds dataset. 
the code has not been tested and/or executed on other datasets, where some inconsistency in loading data exists for example for the imagenet.

to run and generate embeddings for ft data it is important to set -f, or --load_ftmodel = True
- python precompute_activations.py -d waterbirds -m ViT-B~32 -s -f (train_split, to extract training embedings)
- python precompute_activations.py -d waterbirds -m ViT-B~32  -f
- python msae/precompute_activations.py --ft_manifest results/waterbirds/clip_ft_BIASED_400_.../ft_train_manifest.csv  (fine_tune set activation generation)




#### 
# to fine tune on balanced dataset and then bias the high performing finetuned models
####
# ViT-B/32, from the balanced-200 checkpoint
python run_waterbirds_msae.py \
  --bias_balanced_ft results/waterbirds/sweep_20260709_124303/clip_ft_BALANCED_200_20260709_124828/model.pt \
  --max_samples 200 --ft_epochs 3 \
  --output_dir results/waterbirds/twostage_vitb32

# ViT-L/14, from the balanced-200 checkpoint
python run_waterbirds_msae.py \
  --bias_balanced_ft results/waterbirds/sweep_20260709_135925/clip_ft_BALANCED_200_20260710_092743/model.pt \
  --max_samples 200 --ft_epochs 3 \
  --output_dir results/waterbirds/twostage_vit14

##### Now, start MSAE training process #####

precompute_activations.py will generate and save embeddings. Then, we need to run it for the existing vocabulary to get their embeddings as well. 


python msae/precompute_activations.py -d laion_unigram -m ViT-B~32
python msae/precompute_activations.py -d laion_bigrams -m ViT-B~32
python msae/precompute_activations.py -d disect -m ViT-B~32

run the above command for all vocab file names in vocab directory
##### for two-staged fine tuned, both models, waterbird specialized dataset
# ── ViT-B/32 (two-stage) ──
.clip-env/bin/python msae/precompute_activations.py 
  -d waterbirds_domain -m ViT-B~32 -f 
  --ft_run_dir results/waterbirds/twostage_vitb32_100_400/clip_ft_BIASEDfromBALANCED_400_20260721_103542 
  --max_samples 400 
  --output_dir results/waterbirds/twostage_vitb32_100_400/embeddings



output: waterbirds_domain_ViT-B~32_ft400_-1_text_793_512.nppy
waterbirds_domain_ViT-B~32_ft400_-1_text_793_512.txt
# ── ViT-L/14 (two-stage) ──
.clip-env/bin/python msae/precompute_activations.py 
  -d waterbirds_domain -m ViT-L~14 -f 
  --ft_run_dir results/waterbirds/twostage_vit14/clip_ft_BIASEDfromBALANCED_400_20260720_154809 
  --max_samples 400 
  --output_dir results/waterbirds/twostage_vit14/embeddings

#####


After having all npy files for train/test and text vocabularies, we can start training the sae model:
-dt: train_set
-ds: test_set
-dm: other modality

--zeroshot
python msae/train.py -dt results/waterbirds/embeddings/waterbirds_ViT-B~32_train_image_4795_512.npy        -ds results/waterbirds/embeddings/waterbirds_ViT-B~32_test_image_4795_512.npy        -dm results/waterbirds/embeddings/waterbirds_ViT-B~32_test_text_5794_512.npy        --expansion_factor 21 --epochs 20 -m ReLUSAE -a ReLU_01


--finetuned
1- RelUSAE 
python msae/train.py -dt results/waterbirds/embeddings/waterbirds_ViT-B~32_fttrain_image_762_512.npy        -ds results/waterbirds/embeddings/waterbirds_ViT-B~32_ft400_test_image_5794_512.npy        -dm results/waterbirds/embeddings/waterbirds_ViT-B~32_ft400_test_text_5794_512.npy        --expansion_factor 32 --epochs 20 -m ReLUSAE -a ReLU_01


python msae/train.py -dt results/waterbirds/embeddings/waterbirds_ViT-B~32_fttrain_image_762_512.npy        -ds results/waterbirds/embeddings/waterbirds_ViT-B~32_ft400_test_image_5794_512.npy        -dm results/waterbirds/embeddings/waterbirds_ViT-B~32_ft400_test_text_5794_512.npy        --expansion_factor 32 --epochs 20 -m ReLUSAE -a ReLU_03


2- TopKSAE
python msae/train.py -dt results/waterbirds/embeddings/waterbirds_ViT-B~32_fttrain_image_762_512.npy        -ds results/waterbirds/embeddings/waterbirds_ViT-B~32_ft400_test_image_5794_512.npy        -dm results/waterbirds/embeddings/waterbirds_ViT-B~32_ft400_test_text_5794_512.npy        --expansion_factor 8 --epochs 20 -m TopKSAE -a TopKReLU_256

python msae/train.py -dt results/waterbirds/embeddings/waterbirds_ViT-B~32_fttrain_image_762_512.npy        -ds results/waterbirds/embeddings/waterbirds_ViT-B~32_ft400_test_image_5794_512.npy        -dm results/waterbirds/embeddings/waterbirds_ViT-B~32_ft400_test_text_5794_512.npy        --expansion_factor 8 --epochs 20 -m TopKSAE -a TopKReLU_64

3- MSAE_UW
python msae/train.py -dt results/waterbirds/embeddings/waterbirds_ViT-B~32_fttrain_image_762_512.npy        -ds results/waterbirds/embeddings/waterbirds_ViT-B~32_ft400_test_image_5794_512.npy        -dm results/waterbirds/embeddings/waterbirds_ViT-B~32_ft400_test_text_5794_512.npy        --expansion_factor 8 --epochs 20 -m MSAE_UW -a ""


4- MSAE_RW
python msae/train.py -dt results/waterbirds/embeddings/waterbirds_ViT-B~32_fttrain_image_762_512.npy        -ds results/waterbirds/embeddings/waterbirds_ViT-B~32_ft400_test_image_5794_512.npy        -dm results/waterbirds/embeddings/waterbirds_ViT-B~32_ft400_test_text_5794_512.npy        --expansion_factor 8 --epochs 20 -m MSAE_RW -a ""


outouts: it saves sae_weights


(didn't go through the details yet, just adjust input according to the commands existing in their repository)

after training the model, it will be saved as a file with name like this:

(../16384_512_ReLU_01_False_False_0.0_waterbirds_ViT-B~32_train_image_4795_512.pth)
##### 
to run for two-staged ft (bcs of conflict in importing from inside and outside of the moduls, just we need to run this)
# ── ViT-B/32 two-stage (produces the 4096_512_TopKReLU_256_RW... SAE) ──
.clip-env/bin/python tools/run_msae_module.py msae.train \
  -dt results/waterbirds/twostage_vitb32_100_400/embeddings/waterbirds_ViT-B~32_ft400_fttrain_image_762_512.npy \
  -ds results/waterbirds/twostage_vitb32_100_400/embeddings/waterbirds_ViT-B~32_ft400_train_image_4795_512.npy \
  -dm results/waterbirds/twostage_vitb32_100_400/embeddings/waterbirds_ViT-B~32_ft400_train_text_4795_512.npy \
  --expansion_factor 8 --epochs 20 -m MSAE_RW -a ""

# ── ViT-L/14 two-stage (produces the 6144_768_TopKReLU_256_RW... SAE) ──
.clip-env/bin/python tools/run_msae_module.py msae.train \
  -dt results/waterbirds/twostage_vit14/embeddings/waterbirds_ViT-L~14_ft400_fttrain_image_762_768.npy \
  -ds results/waterbirds/twostage_vit14/embeddings/waterbirds_ViT-L~14_ft400_train_image_4795_768.npy \
  -dm results/waterbirds/twostage_vit14/embeddings/waterbirds_ViT-L~14_ft400_train_text_4795_768.npy \
  --expansion_factor 8 --epochs 20 -m MSAE_RW -a ""
#####


For concept naming we need to run sae_naming script correctly, before msea_ftclip or inside it
example:

python msae/sae_naming.py -m results/waterbirds_ViT-B~32_fttrain_image_762_512/sae_weights/16384_512_TopK_64_RW_False_False_0.0_waterbirds_ViT-B~32_fttrain_image_762_512.pth  -v results/waterbirds/embeddings/disect_ViT-B~32_ft400_-1_text_20000_512.npy -p results/waterbirds/embeddings/concept_match/MSAE_RW/

python msae/sae_naming.py -m results/waterbirds_ViT-B~32_fttrain_image_762_512/sae_weights/16384_512_TopKReLU_256_RW_False_False_0.0_waterbirds_ViT-B~32_fttrain_image_762_512.pth  -v results/waterbirds/embeddings/disect_ViT-B~32_ft400_-1_text_20000_512.npy -p results/waterbirds/embeddings/concept_match/MSAE_RW/

python msae/sae_naming.py -m results/waterbirds_ViT-B~32_fttrain_image_762_512/sae_weights/16384_512_TopK_64_UW_False_False_0.0_waterbirds_ViT-B~32_fttrain_image_762_512.pth  -v results/waterbirds/embeddings/disect_ViT-B~32_ft400_-1_text_20000_512.npy -p results/waterbirds/embeddings/concept_match/MSAE_UW/

python msae/sae_naming.py -m results/waterbirds_ViT-B~32_fttrain_image_762_512/sae_weights/16384_512_ReLU_01_False_False_0.0_waterbirds_ViT-B~32_fttrain_image_762_512.pth  -v results/waterbirds/embeddings/disect_ViT-B~32_ft400_-1_text_20000_512.npy -p results/waterbirds/embeddings/concept_match/ReLUSAE
/ReLU_01/

python msae/sae_naming.py -m results/waterbirds_ViT-B~32_fttrain_image_762_512/sae_weights/16384_512_ReLU_03_False_False_0.0_waterbirds_ViT-B~32_fttrain_image_762_512.pth  -v results/waterbirds/embeddings/disect_ViT-B~32_ft400_-1_text_20000_512.npy -p results/waterbirds/embeddings/concept_match/ReLUSAE
/ReLU_03/

python msae/sae_naming.py -m results/waterbirds_ViT-B~32_fttrain_image_762_512/sae_weights/16384_512_TopKReLU_64_False_False_0.0_waterbirds_ViT-B~32_fttrain_image_762_512.pth  -v results/waterbirds/embeddings/disect_ViT-B~32_ft400_-1_text_20000_512.npy -p results/waterbirds/embeddings/concept_match/TopKSAE/TopKReLU_64/

python msae/sae_naming.py -m results/waterbirds_ViT-B~32_fttrain_image_762_512/sae_weights/16384_512_TopKReLU_256_False_False_0.0_waterbirds_ViT-B~32_fttrain_image_762_512.pth  -v results/waterbirds/embeddings/disect_ViT-B~32_ft400_-1_text_20000_512.npy -p results/waterbirds/embeddings/concept_match/TopKSAE/TopKReLU_256/

####
for two-stage fine-tunning:

####

output: concept_match_score in /embeddings/concept_match/model/activation(if any) 
it will be loaded and used in msae_ftclip

*****

Then, go to msae_ftclip and run final concept extraction and experiments

next step:
1- do the same experiments for instances in the training set
2- find concepts existing in misclassified images, separated into 2 nonmatching group, then look for them in the training images in relative group

