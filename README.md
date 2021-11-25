# Augmented Random Forest
    Simple Post-Training Robustness Using Test Time Augmentations and Random Forest
    This repo reproduces all the reuslts shown in our paper.

# Init project

    1) Run in the project dir: source ./init_project.sh
    2) Create validation set and test set indices for all dataset by running:
    python src/scripts/set_val_test_inds.py
    This generates the 'test' and 'test-val' subsets (as explained in the paper) for each dataset

# Train
    Train Resnet networks for cifar10, cifar100, svhn, and tiny_imagenet using src/train.py.
    
    For example, for CIFAR-10 run:
    1) Regular network: python src/scripts/train.py --dataset cifar10 --net resnet34 --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00
    2) TRADES: python src/scripts/train.py --dataset cifar10 --net resnet34 --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/adv_robust_trades --adv_trades True
    3) VAT: python src/scripts/train.py --dataset cifar10 --net resnet34 --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/adv_robust_vat --adv_vat True

    If you wish also to reproduce results for the ensemble, train 9 more networks in: 
    /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_01
    /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_02
    /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_03
    /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_04
    /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_05
    /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_06
    /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_07
    /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_08
    /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_09

# Attack
    For attacking a network, use src/attack.py.
    
    For example, to attack CIFAR-10 with the FGSM^2 attack (defined in the paper), run:
    python src/scripts/attack.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00 --attack fgsm --targeted True --attack_dir fgsm2 --eps 0.031

    Prior to training the Random Forest classifier, one has to generate all the non-adapted attacks in section 4 in the paper: fgsm1, fgsm2, jsma, pgd1, pgd2, cw, cw_Linf, square, and boundary. The complete set of attacks one must run is given here:
    1) [FGSM^1]: python src/scripts/attack.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00 --attack fgsm --targeted True --attack_dir fgsm1 --eps 0.01
    2) [FGSM^2]: python src/scripts/attack.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00 --attack fgsm --targeted True --attack_dir fgsm2 --eps 0.031
    3) [JSMA]: python src/scripts/attack.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00 --attack jsma --targeted True --attack_dir jsma
    4) [PGD^1]: python src/scripts/attack.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00 --attack pgd --targeted True --attack_dir pgd1 --eps 0.01 --eps_step 0.003
    5) [PGD^2]: python src/scripts/attack.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00 --attack pgd --targeted True --attack_dir pgd2 --eps 0.031 --eps_step 0.003
    6) [Deepfool]: python src/scripts/attack.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00 --attack deepfool --targeted False --attack_dir deepfool
    7) [CW_L2]: python src/scripts/attack.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00 --attack cw --targeted True --attack_dir cw
    8) [CW_Linf]: python src/scripts/attack.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00 --attack cw_Linf --targeted True --attack_dir cw_Linf --eps 0.031
    9) [CW_Linf]: python src/scripts/attack.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00 --attack square --targeted False --attack_dir square --eps 0.031
    10) [Boundary]: python src/scripts/attack.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00 --attack boundary --targeted True --attack_dir boundary


# Fit the random forest
    After attacking a network with the above 10 attack, train the random forest by running:
    python src/scripts/train_random_forest.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00
    
    The random forest parameters will be saved under:
    /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00/random_forest/random_forest_classifier.pkl
    

# Adaptive white-box BPDA attack:
    After saving the random forest weights, you can attack the ARF defense.
    
    1) First, create a substitute model using:
    python src/scripts/train_random_forest_sub.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00
    
    2) Second, call the BPDA attack:
    python src/scripts/attack.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00 --attack bpda --targeted True --eps 0.031 --eps_step 0.007 --max_iters 10
    
# Evaluation
    Use src/scrips/eval.py to evaluate the defenses.
    For evaluation a plain model without any defense, run:
    python src/scripts/eval.py --checkpoint_dir /tmp/adversarial_robustness/cifar10/resnet34/regular/resnet34_00 --method simple --attack_dir <YOUR_SELECTED_ATTACK> --dump_dir simple
    
    For calculating accuracy on the Ensemble, TTA, or ARF, replace the "simple" above with "ensemble", "tta", or "random_forest", respectively.

