[0 .. 4] Hi, I'm pi-0.5, I'm the model trained to run on this robot by the team

[5-7] "Sfogliatelle" (with stars)

[7-10] to detect anomalies in my trail of lego bricks, which has to be PERFECT. 

[10-13] I'm really simple, I can only do two things: move a brick close to a camera, and put it in a yellow or orange tray. (text: pick the brick, put the brick in the yellow tray) 

[14-17] ...ok, maybe three. (yellow splits, "orange" appears beneath, video where the orange tray appears)

[17-20] or four ("the brick" becomes ANY brick)

[20-30] or more (unicorn gets put in the middle as distractor, bricks get dropped on the trays everywhere). Video of the whole process from end to finish with red "REC" appearing with no editing.

[30-45] I autonomously do anomaly detection using the AnomalyLib, by training a model on a few samples of our objects < slide / written: ~5/10 samples or whatever it is> < doodled: lego bricks> and using some really cool models (names) to detect healthy bricks < side-by-side view of the RGB video feed and the image with fault detection) from defective ones < while saying defective, the video feed shows a person drawing an angry face on the brick and putting it back there >,  < on the right, picture of the anomaly>



##### non so dove mettere 0

Training was performed with Physical Ai Studio, Intel's tool to visualize robots, handle cameras, and train tasks from end to end. We recorded a dataset of 156 episodes which was used to finetune a pretrained pi-0.5, which with the help of Modal could be trained on some conveniently lent top-tier hardware via multiple hours of work.



##### non so dove mettere 1

[45-1:00] All of this is running in real time on the new intel core x9 388h, featuring a B390 iGPU with 122 TOPS and an NPU (int8) of 50 TFLOP.

< comparison tra Pi0.5 sulla iGPU e sulla NPU, stessa cosa con l'anomaly detection tra iGPU/CPU e NPU >

< it was a no brainer to run the VLA model to better run on the iGPU, achieving inference of xxx with xxx wattss and xxx termals, while anomalylib could run on the NPU with some tweaking achieving xxx and xxx xxx. >

< that's it? No! We further compiled the model to fit the new intel platform, achieving a xxx speedup over the standard OpenVINO optimization obtained by Physical AI Studio. >



##### non so dove mettere 2

< change of voice to a male one>

[x...] Training was done by recording x episodes of actions performed by the robot, which in our case are 



##### non so dove mettere 3

[x] on top of that, we - "vibecoded"

[x..] < vibecoded gets striken through, writes "meticulously prompted" 

[...] < quick video of a really long Claude prompt showing how we built an interface for the anomaly detection pipeline >

[...] BUILT a pretty dashboard to display status, results and promptly alert the user when there's an anomaly (comically loud alert). 

< idea: image of claude (digital) being pushed away from the screen by the robot >


