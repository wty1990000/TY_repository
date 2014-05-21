#include <iostream>
#include <vector>
#include "fftw3.h"
#include "EasyBMP.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "CUDA_COMP.cuh"

//Define the parameters used int the DIC simulation.
const int iMarginX = 10,	iMarginY = 10;
const int iGridX = 10,		iGridY = 10;
const int iSubsetX = 16,	iSubsetY =16;
const int iMaxIteration = 20;
const float fDeltaP = 0.001f;

//Utility functions Declaration in processing the BMP images
bool LoadBMPs(BMP& Image1, BMP& Image2, int argc, char** argv);
bool LoadBmpsToArray(const BMP& Image1, const BMP& Image2, std::vector<float>& Img1, std::vector<float>& Img2);


int main(int argc, char** argv)
{
	using namespace std;

	//Get the image from command line
	cout<<endl
			<<"----------GPU Digital Image Correlation----------"<<endl
			<<"-----Usage: PI_DIC.exe image1.bmp image2.bmp-----"<<endl;
	if(argc < 3){
		cout<<endl
			<<"Cannot parse from the command line parameter: Please follow the Usage."
			<<"----------GPU Digital Image Correlation----------"<<endl
			<<"-----Usage: PI_DIC.exe image1.bmp image2.bmp-----"<<endl;
		exit(0);
	}
	
	//Load the two images and get the arrays of intensity values.
	BMP Image1, Image2;			//Images managed by BMP class
	if(!LoadBMPs(Image2, Image2, argc, argv)){
		cout<<"Unable to load the images. Please try again."<<endl;
		exit(0);
	}		
	vector<float> Img1, Img2;	//Used for storing the newly generated images with border 2 pixels.
	if(!LoadBmpsToArray(Image1, Image2, Img1, Img2)){
		cout<<"Error! The array size of two input images are different! Check the BMP images"<<endl;
		exit(0);
	}
	else
		cout<<"Images are successfully loaded into arrays. Starting computation..."<<endl;

	/*------------------------------Real computation starts here--------------------------------
	  Totally, there are three steps:
	  1. Precomputation of images' gradients matrix and bicubic interpolation matrix
	  2. Using FFT to transform the two images into frequency domain, and after per-
	  forming ZNCC, transforming the results back.
	  3. A Gaussian Newton's optimization method is used to estimate the warped images.
	*/

	//Step 1: GPU pre-computation for the images' gradients and the bicubic interpolation matrix 
	float *hptr_Img1 = Img1.data();
	float *hptr_Img2 = Img2.data();


	
	return 0;
}

//Utility functions implementation
bool LoadBMPs(BMP& Image1, BMP& Image2, int argc, char** argv)
{
	if(Image1.ReadFromFile(argv[1]) && Image2.ReadFromFile(argv[2])){
		if(Image1.TellHeight() != Image2.TellHeight() ||
			Image1.TellWidth() != Image2.TellWidth()){
				std::cout<<"Error! The scale of the two input images should be identical!"<<std::endl;
				return FALSE;
		}
		return TRUE;
	}
	else
		return FALSE;
}
bool LoadBmpsToArray(const BMP& Image1, const BMP& Image2, std::vector<float>& Img1, std::vector<float>& Img2)
{
	RGBApixel pixelValue;
	for(int row=0; row<Image1.TellHeight; row++){
		for(int col=0; col<Image1.TellWidth; col++){
			pixelValue = Image1.GetPixel(col,row);
			Img1.push_back(float((int)pixelValue.Red + (int)pixelValue.Green + (int)pixelValue.Blue)/3.0f);
			pixelValue = Image2.GetPixel(col,row);
			Img2.push_back(float((int)pixelValue.Red + (int)pixelValue.Green + (int)pixelValue.Blue)/3.0f); 
		}
	}
	if(Img1.size() == Img2.size()){
		return TRUE;
	}
	else
		return FALSE;
}
