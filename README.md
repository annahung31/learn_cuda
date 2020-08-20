
### Notes
See some [notes](lecture_notes.md) of [平行程式](https://youtu.be/t_q0Tajpyso)


### Case study

[Case 1](case_study_1.cu): Cupy variables onto/off GPU.  
[Case 2](case_study_2.cu): How to index the threads.  
[Case 3](func_1.cu): `atomicAdd(result, i)` means `result += i`  


### Compile the code
```
nvcc -o kernel kernel.cu
```



# Reference
* A simple tutorial made in 2013: https://www.youtube.com/watch?v=_41LCMFpsFs


* https://www.youtube.com/watch?v=Ed_h2km0liI&list=PL5B692fm6--vScfBaxgY89IRWFzDt0Khm&index=24&t=0s 

* https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/ 

* 中文 tutorial: https://kheresy.wordpress.com/2008/01/15/nvidia-cuda-%e7%9b%b8%e9%97%9c%e6%96%87%e7%ab%a0%e7%9b%ae%e9%8c%84/ 

* 中文教學pdf: http://epaper.gotop.com.tw/pdf/ACL031800.pdf 

* NTHU 線上課程： 周志遠老師的[平行程式](https://youtu.be/t_q0Tajpyso) 從第15講開始

* PyTorch [C++ distribution](https://pytorch.org/cppdocs/installing.html)