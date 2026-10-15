# llama-1b-lab
A hands-on, modular repo to pretraining, scaling, and aligning a 1B Llama model from scratch.


48gb A40
NCCL_P2P_DISABLE=1 python -m ddp_05.naive_ddp_debug
DDPs run comfortable with m size model
but runs only to some steps in l size model 
loss after 100 epoch
s 1.9 loss 
m 2.01
l Step: 99, loss: 2.76701

FSDP:
TINY MODEL
ls = Step: 99, loss: 6.62315

Kill process
pkill -9 -f "multiprocessing.spawn"
pkill -9 -f naive_ddp

Git sshkey
git remote set-url origin git@github.com:SamipThulung/llama-1b-lab.git
git config --global user.email "samipthulung3@gmail.com"
git config --global user.name "SamipThulung"
ssh-keygen -t ed25519 -C "samipthulung3@gmail.com"
cat ~/.ssh/id_ed25519.pub

git push origin main

Fetch GitHub's host key into known_hosts
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts

Verifying-finger print
ssh-keygen -lf ~/.ssh/known_hosts
Test Authentication
ssh -T git@github.com

echo "filename.txt" >> .gitignore

Initial run megatron
pip install -r requirements.txt
python -m dataset_03.megatron_data
torchrun --nproc_per_node=2 -m megatron_07.megatron_train
NCCL_P2P_DISABLE=1 python -m megatron_07.megatron_train # for l4 gpu
niteration   99 | lr 3.01e-05 | loss 1.8429 | grad_norm 0.5999701343494377 | step_ok True
n iteration   99 | lr 3.01e-05 | loss 1.8429 | grad_norm 0.5999701343494377 | step_ok True
s iteration   99 | lr 3.01e-05 | loss 1.7891 | grad_norm 0.7646029131164056 | step_ok True
s iteration  199 | lr 3.00e-05 | loss 1.7078 | grad_norm 0.7362759113771518 | step_ok True
iteration  999 | lr 3.00e-05 | loss 1.4359 | grad_norm 0.7253650929893452 | step_ok True

m iteration   99 | lr 3.01e-05 | loss 1.8072 | grad_norm 0.8433550864141988 | step_ok True
miteration 1999 | lr 3.00e-05 | loss 1.3523 | grad_norm 0.7049172342023569 | step_ok True


l iteration   99 | lr 3.01e-05 | loss 1.9120 | grad_norm 1.2634242668657127 | step_ok True

Generated output:
There is a lot of things that I can do with my own. I have a lot of fun and I can’t get to know what I’m doing. I

l-model:
********************************
GPU 0 parameters: 393,473,280
Full model parameters: 786,853,120
Model size: 786.85M
********************************

Painless way GCP
curl https://sdk.cloud.google.com | bash
Restart. 
gcloud auth login
gcloud config set project llms-511402
gcloud storage cp -r megatron_07/checkpoints_l/step_00010000 gs://llm_model_samip

********************************
GPU 0 parameters: 393,473,280
Full model parameters: 786,853,120
Model size: 786.85M
********************************

============================================================
PROMPT: Once upon a time, there was a young boy who
RESPONSE:  was given two humpcells by the teacher. He then discovered that from the time he was 7, his sound was normal. I am now 8 or 10. He may have been as good as Julius Caesar but behind his inactivity, I discovered that it wasn't really a time for him to be over the hill.
How did his journey from October 1940 to the present time mean that John Stuart had to travel through the late 70's and early 90's. In the beginning, he found that he had gone through meditation with his children and worked with them through the ups and downs of the 90's. In 1957, he was diagnosed with depression and a certain type of anxiety.
The following day, he returned to the old world and found himself lost in a situation called "God's'reyxia" which led him to the end of an attack.
I had the second chance to join the old world and take a lesson with him. I was looking to meet John and Jene there and there and he was the coolest person. My brain was growing. I could finally figure out where he was.
He was the first teacher in my life, who was always the youngest person in my family. I had learned that I was going through a difficult time. I did not have that for myself, but there were a lot of good riverheads and so many little students on the teachers. I was there for that and I felt like I needed to go through my final year. I envied the school and would always find someone I could trust.
The first year of school, I stayed in Cabela High School which was my first year. I was so grateful to the teachers and the people of Cabela High (they were my only child), and I couldn't stop being there. I was pretty lucky to be here… I went to school when my older child had started the Practice! I only felt like I was growing up while in high school. I had a great, unique, high school. I love the feeling of being in those higher buildings.
What when I feel that I’m not as good as I was when I was 13 or 16? Well, I had the after school. I did not want to have children because I didn’t have that kind of opportunity. I wanted to have my own life that I had no heart or will I ever be able to live an active life and I knew I was going to be in my commencement year… I looked
TEMPERATURE: 0.8
GENERATED TOKENS: 512
============================================================

============================================================
PROMPT: The sun was setting over the mountains, and
RESPONSE:  the moon started to rock our path and keep on playing. The hills were overflowing and the sun was still shining. The sound of the cloud just out of tune shot the sky and looked like we had just watched a TV series. The wind was in the sky and the rain was dripping in the background. The wind left up on the horizon to a closed and dark sky. The next morning, the wind stopped to dry as the wind came over the horizon and the wind became a lot of “difficult” and great. The wind just as moving as it poked down and the wind picked up. The wind flowed into the sky, and then came the sound of the wind from the wind. The wind started to shaking. The wind rode down and the winds cleared to the ground. The wind had started to slip to the ground. The wind on the ground genuinely popped up and exploded. We didn’t forget the light of the wind coming from the wind.
The wind took a long time, as the wind slowly tickled the wind and the wind. The wind and the wind were always around. The wind was not Division 32, but it was a newly acquired building called Kuhnst. The building belongs to the Kuhnst and a way of life for Kuhnst. The building is a lovely building with a lot of history.
The building is an old building. The building has been in Kuhnst for years. The building is a short walk from the Kuhnst building. The building is a little steep and has a beautiful view. There is also a short walk to the west. The building runs through the neighbourhood. The building has been placed inside the building for Kuhnst since 1974. The building has been used by the Kuhnst family since 2001. The building has been a beautiful building for Kuhnst. We didn’t have a lot of time to build this building. The building has been very beautiful that we still have a lot of history.
The building is a private building. The building is a traditional building of the famous buildings built in 2015. The building is about a 5 minute walk from the Kuhnst building. The building has been close to the building since 2015.
The building is about a 5 minute walk to the east. The building just beyond the building is a 5 minute walk to the south. The building is as good as you can get. The building has
TEMPERATURE: 0.8
GENERATED TOKENS: 512
============================================================

============================================================
PROMPT: The largest planet in our solar system is
RESPONSE:  the Arctic full moon, the planet in largest solar system, the largest solar system in the world. The planet is the number one global windmills
- Learning and learning about the world, or by studying in different areas of the planet, and learning about the world frequently, and by learning about different different basic structure and shape of the solar system.
- Teaching students on thetech and smart technologies, improve the efficiency of life and an awareness about the future of the world. Blog
- More than 100 young people, and many more in the private sector, are turning to the HTML5 platform.
- Fewer communities are able to use HTML5 as a backend for their education, and it's a good way to look for potential clients as well.
- Reading and learning about the world can be challenging, but it's not entirely clear at all.
- The default approach of the HTML5 platform is to work with all types of plugs: HTML5, HTML5, PDF5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5, HTML5
- Notes are taken from the "Code"' in the "Library" article: "This is a full list of lectures and lectures." They are available on the "Web" page. Read full profiles at: www.volent.edu.
- The purpose of this website is to provide a means for users to learn about the subject and to learn about the subject and how to use it. An online discussion forum, the "Code" provides information and clips from theIONS of the Web. The discussion forum is also available and available on the "Post" page.
- An information and chatty is available on the "Web" page. The "Web" page is a collection of articles related to the topic and is available on the "Web" page.
- The "Web" website is located at www.unita.weeks.com. A list can be accessed by visiting www.codemasters.org. The "Web's"
TEMPERATURE: 0.8
GENERATED TOKENS: 512
============================================================

============================================================
PROMPT: Mount Everest is located in
RESPONSE:  the lovely and peaceful betterment district of AK-15 2, 514 sqft of cc. resem. with a room and small kitchen which will serve as a standalone kitchen. The kitchen is a great accompaniment to the big day as well as the busy afternoon, and is very well equipped, with a small kitchen and fully charged. The kitchen is great for a relaxing dinner or meal.
|Aven contributed to this article|
TEMPERATURE: 0.8
GENERATED TOKENS: 91
============================================================
