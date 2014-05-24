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
const int iIterationNum = 5;

//Utility functions Declaration in processing the BMP images
int LoadBMPs(BMP& Image1, BMP& Image2, int argc, char** argv);
int LoadBmpsToArray(BMP& Image1, BMP& Image2, std::vector<float>& Img1, std::vector<float>& Img2);


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
	if(0 == LoadBMPs(Image1, Image2, argc, argv)){
		cout<<"Unable to load the images. Please try again."<<endl;
		exit(0);
	}		
	vector<float> Img1, Img2;	//Used for storing the newly generated images with border 2 pixels.
	if(0 == LoadBmpsToArray(Image1, Image2, Img1, Img2)){
		cout<<"Error! The array size of two input images are different! Check the BMP images"<<endl;
		exit(0);
	}
	else
		cout<<"Images are successfully loaded into arrays. Starting computation..."<<endl;

	//Initialize the CUDA runtime library
	cudaInit();

	/*------------------------------Real computation starts here--------------------------------
	  Totally, there are three steps:
	  1. Precomputation of images' gradients matrix and bicubic interpolation matrix
	  2. Using FFT to transform the two images into frequency domain, and after per-
	  forming ZNCC, transforming the results back.
	  3. A Gaussian Newton's optimization method is used to estimate the warped images.
	*/
	int width =  Image1.TellWidth() - 2;
	int height = Image1.TellHeight() -2;
	int iNumberX = int(floor((width - iSubsetX*2 - iMarginX*2)/float(iGridX))) + 1;
	int iNumberY = int(floor((height - iSubsetY*2 - iMarginY*2)/float(iGridY))) + 1;
	int iSubsetW = iSubsetX*2+1;
	int iSubsetH = iSubsetY*2+1;
	int iFFTSubW = iSubsetX*2;
	int iFFTSubH = iSubsetY*2;

	float fTimePrecopmute=0.0f, fTimeFFTCC=0.0f, fTimeICGN=0.0f, fTimeTotal=0.0f;

	float *fZNCC = (float*)malloc(iNumberX*iNumberY*sizeof(float));
	float *fdP	 = (float*)malloc(iNumberX*iNumberY*6*sizeof(float));
	float *fdPXY = (float*)malloc(iNumberX*iNumberY*2*sizeof(float));
	int *iU = (int*)malloc(iNumberX*iNumberY*sizeof(float));
	int *iV = (int*)malloc(iNumberX*iNumberY*sizeof(float));

	for(int i=0; i<iNumberY; i++){
		for(int j=0; j<iNumberX; j++){
			fdPXY[(i*iNumberX+j)*2+0] = float(iMarginX + iSubsetY + i*iGridY);
			fdPXY[(i*iNumberX+j)*2+1] = float(iMarginY + iSubsetX + j*iGridX);
		}
	}

	combined_function(Img1,Img2,fdPXY,width,height,
		iSubsetH,iSubsetW,iSubsetX,iSubsetY,iNumberX,iNumberY,iFFTSubH,iFFTSubW,iIterationNum,fDeltaP,
		iU,iV,fZNCC,fdP,fTimePrecopmute,fTimeFFTCC,fTimeICGN,fTimeTotal);

	ofstream OutputFile;
	OutputFile.open("Results.txt");
	for(int i =0; i<iNumberY; i++){
		for(int j=0; j<iNumberX; j++){
			OutputFile<<int(fdPXY[(i*iNumberX+j)*2+1])<<", "<<int(fdPXY[(i*iNumberX+j)*2+0])<<", "<<iU[i*iNumberX+j]<<", "
				<<fdP[(i*iNumberX+j)*6+0]<<", "<<fdP[(i*iNumberX+j)*6+1]<<", "<<fdP[(i*iNumberX+j)*6+2]<<", "<<fdP[(i*iNumberX+j)*6+3]<<", "<<iV[i*iNumberX+j]<<", "<<fdP[(i*iNumberX+j)*6+4]<<", "<<fdP[(i*iNumberX+j)*6+5]<<", "
				<<fZNCC[i*iNumberX+j]<<endl;
		}
	}
	OutputFile.close();	

	OutputFile.open("Time.txt");
	OutputFile << "Interval (X-axis): " << iGridX << " [pixel]" << endl;
	OutputFile << "Interval (Y-axis): " << iGridY << " [pixel]" << endl;
	OutputFile << "Number of POI: " << iNumberY*iNumberX << " = " << iNumberX << " X " << iNumberY << endl;
	OutputFile << "Subset dimension: " << iSubsetW << "x" << iSubsetH << " pixels" << endl;
	OutputFile << "Time comsumed: " << fTimeTotal << " [millisec]" << endl;
	OutputFile << "Time for Pre-computation: " << fTimePrecopmute << " [millisec]" << endl;
	OutputFile << "Time for integral-pixel registration: " << fTimeFFTCC / (iNumberY*iNumberX) << " [millisec]" << endl;
	OutputFile << "Time for sub-pixel registration: " << fTimeICGN / (iNumberY*iNumberX) << " [millisec]" << " for average iteration steps of " << float(iIterationNum) / (iNumberY*iNumberX) << endl;
	OutputFile << width << ", " << height << ", " << iGridX << ", " << iGridY << ", " << endl;

	OutputFile <<"Time for computing every FFT:"<<fTimeFFTCC<<"[miliseconds]"<<endl;
	OutputFile <<"Time for ICGN:"<<fTimeICGN<<endl;

	OutputFile.close();
	
	free(fZNCC);
	free(fdP);
	free(fdPXY);
	free(iU);
	free(iV);

	return 0;
}

//Utility functions implementation
int LoadBMPs(BMP& Image1, BMP& Image2, int argc, char** argv)
/*Input: Command line BMP files in argv[][]
 Output: BMP images saved in BMP class objects
Purpose: Load BMP images.
*/
{
	if(Image1.ReadFromFile(argv[1]) && Image2.ReadFromFile(argv[2])){
		if(Image1.TellHeight() != Image2.TellHeight() ||
			Image1.TellWidth() != Image2.TellWidth()){
				std::cout<<"Error! The scale of the two input images should be identical!"<<std::endl;
				return 0;
		}
		return 1;
	}
	else
		return 0;
}
int LoadBmpsToArray(BMP& Image1, BMP& Image2, std::vector<float>& Img1, std::vector<float>& Img2)
/*Input: Two BMP class objects that contain images' information.
 Output: Two vector's that contain the gray level intensity values of the two images
Purpose: Construct the arrays for the two images' intensity values. 
*/
{
	RGBApixel pixelValue;
	for(int row=0; row<Image1.TellHeight(); row++){
		for(int col=0; col<Image1.TellWidth(); col++){
			pixelValue = Image1.GetPixel(col,row);
			Img1.push_back(float((int)pixelValue.Red + (int)pixelValue.Green + (int)pixelValue.Blue)/3.0f);
			pixelValue = Image2.GetPixel(col,row);
			Img2.push_back(float((int)pixelValue.Red + (int)pixelValue.Green + (int)pixelValue.Blue)/3.0f); 
		}
	}
	if(Img1.size() == Img2.size()){
		return 1;
	}
	else
		return 0;
}
