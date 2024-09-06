
# nanogpt-lecture

Code created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT. Publishing here as a Github repo so people can easily hack it, walk through the `git log` history of it, etc.

NOTE: sadly I did not go too much into model initialization in the video lecture, but it is quite important for good performance. The current code will train and work fine, but its convergence is slower because it starts off in a not great spot in the weight space. Please see [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) for `# init all weights` comment, and especially how it calls the `_init_weights` function. Even more sadly, the code in this repo is a bit different in how it names and stores the various modules, so it's not possible to directly copy paste this code here. My current plan is to publish a supplementary video lecture and cover these parts, then I will also push the exact code changes to this repo. For now I'm keeping it as is so it is almost exactly what we actually covered in the video.

### License

MIT

# Completions

### Wikipedia pre-training
```
Graphics processing unit:
The Graphics Processing Unit (GPU) is a Graphics-core graphics-core processor designed for ICM-C products, featuring the GPU and routing processors and IDEs, alongside the GROM/C 3100 language. It served out through, syntax, and precedent GPUs, developed by IBM ensuring features like continuous-time device quality and effective modality of JSON processors. With open-source seamless code readability, Trade has become a vital component in computer software development tools, developed in 2000 by the Reemann and Reemann Alliance. This innovative approach allows users to effortlessly manage flood devices without accessing engaged internet configurations. As developers often employ 360-flood 
and 260-flood tools, making warfare accessible and essential for managing and extending products.
```
### Shakespeare fine-tuning
```
MARCIUS:
Nor need not that his father's name;
Where is Coriolanus?

MARCIUS:
Stay not
The duke by the pattern: but I'll stand for
What thou wert so, though they were not in a        
pretent of my name; 'tis a strange piece:
I pray thee, though I thought the testimony, my son,
Where stands above the duke with me! I cannot do    
But withdraw it, and humbly take our vanquishment:  
Be patient, since the lastly king.

GLOUCESTER:
Then whither? stay down with him: where hast        
I loved sight above the common prices?

BUCKINGHAM:
Whereupon, sovereign, my lord, thy lord,--

GLOUCESTER:
And I, my lord.

BUCKINGHAM:
Because my lord, why have heard your lordship       
Spoke saffron, and to be seen so.

GLOUCESTER:
My lord, I know not, because I will pray thee,      
Sunset from thy revenges and duty.

BUCKINGHAM:
Pray, be up thy good lord.

KING HARD III:
'Tis when this deed, thou art not stiff'd;
You shall have an unshapen too late to subject.
Worthy Menenius, that raged Titus,
Unless his action cannot admit him,
Unless his edict take our hatred by mine.
```
