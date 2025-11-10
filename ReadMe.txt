This repo is for MakeCU Hackathon 2025-2026.
MakeCU is a hardware-software hackathon held by Columbia University.
Our Team: CUPu developed a automated robot that is able to sort objects between trash and recyclables.


Inspiration

Environmental sustainability has been a prominent issue in the world. Our team wanted to work on this problem as we believed that it is everyone's duty to work towards a cleaner environment. With this in mind, we wanted to implement a product that could help everybody with sustainability. This led to our idea of making a robot garbage disposal sorter.
What it does

Our robot is able to sense when someone places an object in front of it. After sensing the object it is able to decide whether it is trash or recyclable and move it to its appropriate bin. It makes sustainability more accessible.
How we built it

On the hardware side of our robot, we used a raspberry pi as its controller. This raspberry pi signals to a servo driver that is able move the object into its respective bin. The raspberry pi is also connected to a camera and a ultra sound sensor allowing us to know when an object is in the right location as well as what the object is. The platform where the object is placed and the wall where our electronics are mounted were 3D printed and designed in Solidworks. Our bin and funnel were made from carbon to give it a frame.

On the software side of our robot, our robot utilized a dataset from Kaggle with a multitude of images and their labels. This dataset was used to train our robot to detect whether an object was recyclable or trash.
Challenges we ran into

The first challenge was connecting the RaspberryPi to the laptop - network connectivity took us at least 3 hours. SSH was extremely unreliable with the phone hotspot. Then, training our computer vision model while connected to the phone’s hotspot took a lot longer than expected. Our biggest challenge was making the servomotor rotate based on signals from the raspberry pi. We were initially going to use a servo with a busboard, but we burned the busboard when first connecting it to power. Therefore, we had to redesign our 3D model to fit a new type of servo, and do a lot of re-programing (new libraries, different functions). Lack of reliable batteries available to us Lack of 3D printer accessibility as there were 120 people for 12 3D printers.
Accomplishments that we're proud of

After hours of testing and ensuring the camera was able to detect the items placed in front of it, we were able to make it work and separate the trash from recyclable items based on the data set that we used from Kaggle. Additionally, it was our first time successfully training a model, with a 96% accuracy rate. Furthermore, we are proud of taking a step to tackling a real-world problem that continues to affect everyone. Many of our aspiring engineers went beyond their comfort zone and learned new skills across many domains.
What we learned

We learned how to train a computer vision model using a data set. We learned a lot about hardware: from the basics of connecting microcontrollers to the laptop, and wiring the different components, to furthering our knowledge of embedded systems and design. We also learned about our limits - we tried to stay awake for most of the event, as there were a lot of issues to solve during the project development, but we recognized that some rest was needed in order to preserve our health and deliver the best prototype we could
What's next for P-CuT

Expanding the functionalities to sort the trash into subcategories such as plastic, paper, cans/bottles, trash.
Built With

    gpio0
    numpy
    opencv
    picamera2
    python
    pytorch
    raspberry-pi
    signal
    solidworks


