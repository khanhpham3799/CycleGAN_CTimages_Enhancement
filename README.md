# Welcom to Duy Khanh channel
This is the modified of CycleGAN model for AAPM dataset for CT image enhancement quality from quarter_dose to full_dose images.

## CycleGAN model block diagram: 
![alt text](https://user-images.githubusercontent.com/64471569/142429396-6d255378-4adb-4d95-be1d-1546cbf1b598.png)

## Image preprocessing transformation:
Transform to HU unit and truncated from -1000HU:

    image = 1000*(image-0.0194)/0.0194

    image = np.where(image<-1000,-1000,image)
    
    image = image/4000

## Image enhancement result 
![alt text](https://user-images.githubusercontent.com/64471569/147317421-0484c4a2-8c90-4178-adb2-50df4b125086.png)

![alt text](https://user-images.githubusercontent.com/64471569/147317498-04571336-422f-4ca4-b459-ca666bac677d.png)

![alt text](https://user-images.githubusercontent.com/64471569/147317601-21fa6f74-0f20-494e-9000-deec40bfbd41.png)

![alt text](https://user-images.githubusercontent.com/64471569/147317655-8f6ae53e-5f89-479a-883b-2671b5a6d3b6.png)
