#include <iostream>
#include <vector>
#include "fftw3.h"
#include "EasyBMP.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "PIDIC.cuh"

using namespace std;

int main()
{
	//Load the two images into array of intensity values
	BMP ImageR, ImageT;
	if(!ImageR.ReadFromFile("f0u2.bmp")){
		cout<<"Unable to load Image R. Please try again."<<endl;
		exit(0);
	}
	if(!ImageT.ReadFromFile("f1u2.bmp")){
		cout<<"Unable to load Image T. Please try again."<<endl;
		exit(0);
	}
	if( (ImageR.TellHeight() != ImageT.TellHeight()) ||
		(ImageR.TellWidth()  != ImageT.TellWidth()) )
	{
		cout<<"Error! The scale of the images should be same!."<<endl;
		exit(0);
	}
	float* ImgR = (float*)malloc(ImageR.TellHeight()*ImageT.TellWidth()*sizeof(float));
	float* ImgT = (float*)malloc(ImageT.TellHeight()*ImageT.TellWidth()*sizeof(float));
	RGBApixel pixelValue;
	for(int row=0; row<ImageR.TellHeight(); row++){
		for(int col=0; col<ImageR.TellWidth(); col++){
			pixelValue = ImageR.GetPixel(col,row);
			ImgR[row*ImageR.TellWidth()+col] = float((int)pixelValue.Red + (int)pixelValue.Green + (int)pixelValue.Blue)/3.0f;
			pixelValue = ImageT.GetPixel(col,row);
			ImgT[row*ImageR.TellWidth()+col] = float((int)pixelValue.Red + (int)pixelValue.Green + (int)pixelValue.Blue)/3.0f; 
		}
	}
	
	int iWidth = ImageR.TellWidth(), iHeight = ImageR.TellHeight();

	//Initialize CUDA Runtime Library
	InitCuda();
	//Computed
	computation(ImgR, ImgT, iWidth, iHeight);

	return 0;
}