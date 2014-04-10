
// PiDICDlg.cpp : implementation file
//

#include "stdafx.h"
#include "PiDIC.h"
#include "PiDICDlg.h"
#include "afxdialogex.h"
#include "combination.cuh"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

	// Dialog Data
	enum { IDD = IDD_ABOUTBOX };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};
CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}
void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}
BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()
// CPiDICDlg dialog
CPiDICDlg::CPiDICDlg(CWnd* pParent /*=NULL*/)
: CDialogEx(CPiDICDlg::IDD, pParent)
, m_sRefImgPath(_T(""))
, m_sTarImgPath(_T(""))
, m_iMarginY(0)
, m_iMarginX(0)
, m_iGridSpaceX(0)
, m_iGridSpaceY(0)
, m_iSubsetX(0)
, m_iSubsetY(0)
, m_dNormDeltaP(0)
, m_iMaxIteration(0)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}
void CPiDICDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT1, m_sRefImgPath);
	DDX_Text(pDX, IDC_EDIT5, m_iGridSpaceX);
	DDV_MinMaxInt(pDX, m_iGridSpaceX, 1, 10000);
	DDX_Text(pDX, IDC_EDIT4, m_iGridSpaceY);
	DDV_MinMaxInt(pDX, m_iGridSpaceY, 1, 10000);
	DDX_Text(pDX, IDC_EDIT7, m_iSubsetX);
	DDV_MinMaxInt(pDX, m_iSubsetX, 1, 10000);
	DDX_Text(pDX, IDC_EDIT6, m_iSubsetY);
	DDV_MinMaxInt(pDX, m_iSubsetY, 1, 10000);
	DDX_Text(pDX, IDC_EDIT8, m_dNormDeltaP);
	DDV_MinMaxDouble(pDX, m_dNormDeltaP, 0, 1);
	DDX_Text(pDX, IDC_EDIT9, m_iMaxIteration);
	DDV_MinMaxInt(pDX, m_iMaxIteration, 1, 10000);
	DDX_Text(pDX, IDC_EDIT2, m_sTarImgPath);
	DDX_Text(pDX, IDC_EDITMARGINY, m_iMarginY);
	DDV_MinMaxInt(pDX, m_iMarginY, 1, 10000);
	DDX_Text(pDX, IDC_EDITMARGINX, m_iMarginX);
	DDV_MinMaxInt(pDX, m_iMarginX, 1, 10000);
}
BEGIN_MESSAGE_MAP(CPiDICDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_SelectREfBtn, &CPiDICDlg::OnBnClickedSelectrefbtn)
	ON_BN_CLICKED(IDOK, &CPiDICDlg::OnBnClickedOk)
	ON_BN_CLICKED(IDC_SelectTarBtn, &CPiDICDlg::OnBnClickedSelecttarbtn)
END_MESSAGE_MAP()
// CPiDICDlg message handlers
BOOL CPiDICDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here
	m_iMarginX = 10;
	m_iMarginY = 10;
	m_iGridSpaceX = 10;
	m_iGridSpaceY = 10;
	m_iSubsetX = 16;
	m_iSubsetY = 16;
	m_dNormDeltaP = 0.001;
	m_iMaxIteration = 20;

	UpdateData(FALSE);

	return TRUE;  // return TRUE  unless you set the focus to a control
}
void CPiDICDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}
// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.
void CPiDICDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
	initialize();
}
// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CPiDICDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}
void CPiDICDlg::OnBnClickedSelectrefbtn()
{
	// TODO: Add your control notification handler code here
	CString strFilter;
	CSimpleArray<GUID> aguidFileTypes;
	HRESULT hResult;


	hResult = m_Image1.GetExporterFilterString(strFilter, aguidFileTypes, _T("All Image Files"));
	if (FAILED(hResult))
	{
		MessageBox(NULL, _T("Fail to call GetExporterFilter£¡"), MB_OK);
		return;
	}
	CFileDialog dlg(TRUE, NULL, NULL, OFN_FILEMUSTEXIST, strFilter);
	if (IDOK != dlg.DoModal())
		return;

	m_sFilePath = dlg.GetFolderPath();
	m_sFileName = dlg.GetFileName();
	CString FullPath;
	FullPath = m_sFilePath + _T("\\\\") + m_sFileName;
	m_sRefImgPath = FullPath;
	UpdateData(FALSE);
}
void CPiDICDlg::OnBnClickedSelecttarbtn()
{
	// TODO: Add your control notification handler code here
	CString strFilter;
	CSimpleArray<GUID> aguidFileTypes;
	HRESULT hResult;


	hResult = m_Image2.GetExporterFilterString(strFilter, aguidFileTypes, _T("All Image Files"));
	if (FAILED(hResult))
	{
		MessageBox(NULL, _T("Fail to call GetExporterFilter£¡"), MB_OK);
		return;
	}
	CFileDialog dlg(TRUE, NULL, NULL, OFN_FILEMUSTEXIST, strFilter);
	if (IDOK != dlg.DoModal())
		return;

	m_sFilePath = dlg.GetFolderPath();
	m_sFileName = dlg.GetFileName();
	CString FullPath;
	FullPath = m_sFilePath + _T("\\\\") + m_sFileName;
	m_sTarImgPath = FullPath;
	m_sOutputFilePath = m_sFilePath;
	UpdateData(FALSE);

}
void CPiDICDlg::OnBnClickedOk()
{
	// TODO: Add your control notification handler code here
	// Acquire input parameters
	UpdateData(TRUE);

	// Check if the file path of the reference image and the taget image are set
	if (m_sRefImgPath.IsEmpty()){
		MessageBox(_T("No reference image selected£¡"));
		return;
	}
	if (m_sTarImgPath.IsEmpty()){
		MessageBox(_T("No target image selected£¡"));
		return;
	}
	// Load images using CImage class
	HRESULT hResult = m_Image1.Load(m_sRefImgPath);
	if (FAILED(hResult)){
		MessageBox(_T("Fail to load Reference Image"), NULL, MB_ICONERROR | MB_OK);
		return;
	}
	hResult = m_Image2.Load(m_sTarImgPath);
	if (FAILED(hResult)){
		MessageBox(_T("Fail to load Target Image"), NULL, MB_ICONERROR | MB_OK);
		return;
	}

	// Check if the dimension of the two images are identical
	int m_iImgWidth, m_iImgHeight;
	m_iImgWidth = m_Image1.GetWidth();
	m_iImgHeight = m_Image1.GetHeight();

	if (m_iImgWidth != m_Image2.GetWidth()){
		MessageBox(_T("The width of the two images are not equal!"), NULL, MB_ICONERROR | MB_OK);
		return;
	}
	if (m_iImgWidth != m_Image2.GetWidth()){
		MessageBox(_T("The Heights of the two images are not equal!"), NULL, MB_ICONERROR | MB_OK);
		return;
	}

	int i, j, k, l, m, n; //counter parameters for loop
	LARGE_INTEGER m_Start, m_Stop, m_Freq, m_Begin, m_End, m_Start1, m_Stop1, m_Start2, m_Stop2; //parameters for timer 
	float m_dConsumedTime, m_dPrecomputeTime, m_dFFTTime, m_dICGNTime; //parameters for timer 
	CString m_strMessage; 	//message for pop-out dialog
	m_dConsumedTime = 0;
	m_dPrecomputeTime = 0;
	m_dFFTTime = 0;
	m_dICGNTime = 0;

	//Read images and convert the gray value of intensity to double precision number
	/*----------------Linearlize---------------------*/
	double *m_dImg1 = (double*)malloc(m_iImgHeight*m_iImgWidth*sizeof(double));
	double *m_dImg2 = (double*)malloc(m_iImgHeight*m_iImgWidth*sizeof(double));
	//double **m_dImg1 = new double *[m_iImgHeight];
	//double **m_dImg2 = new double *[m_iImgHeight];
	/*for (i = 0; i < m_iImgHeight; i++){
		m_dImg1[i] = new double[m_iImgWidth];
		m_dImg2[i] = new double[m_iImgWidth];
	}*/

	double m_dTemp, m_dTempX, m_dTempY;
	COLORREF m_PixelColor;

	for (i = 0; i < m_iImgHeight; i++){
		for (j = 0; j < m_iImgWidth; j++){
			m_PixelColor = m_Image1.GetPixel(j, i);
			m_dTemp = double((GetRValue(m_PixelColor) + GetGValue(m_PixelColor) + GetBValue(m_PixelColor)) / 3);
			m_dImg1[i*m_iImgWidth+j] = m_dTemp;
			m_PixelColor = m_Image2.GetPixel(j, i);
			m_dTemp = double((GetRValue(m_PixelColor) + GetGValue(m_PixelColor) + GetBValue(m_PixelColor)) / 3);
			m_dImg2[i*m_iImgWidth+j] = m_dTemp;
		}
	}
	// Pop up a dialog when the two images are successfully loaded
	MessageBox(_T("Images loaded. Start computation."), NULL, MB_OK);

	//-------------All the parameters need to use------------------
	int m_iWidth = m_iImgWidth - 2; // set margin = 1 column
	int m_iHeight = m_iImgHeight - 2; // set margin = 1 row
	//Define the size of subset window for IC-GN algorithm
	int m_iSubsetW = m_iSubsetX * 2 + 1;
	int m_iSubsetH = m_iSubsetY * 2 + 1;
	int m_iBlackSubsetFlag, m_iOutofBoundaryFlag; //Flag parameters
	//Define the size of subset window for FFT-CC algorithm
	int m_iFFTSubW = m_iSubsetX * 2;
	int m_iFFTSubH = m_iSubsetY * 2;
	//Estimate the number of points of interest(POIs)
	int m_iNumberX = int(floor((m_iWidth - m_iSubsetX * 2 - m_iMarginX * 2) / double(m_iGridSpaceX))) + 1;
	int m_iNumberY = int(floor((m_iHeight - m_iSubsetY * 2 - m_iMarginY * 2) / double(m_iGridSpaceY))) + 1;

	//----------------For FFT-CC Inputs/outputs--------------------
	double *m_dZNCC;

	//----------------For ICGN outputs----------------------
	double* m_iU, m_iV;	

	//Initialize the parameters for IC-GN algorithm
	int **m_iIterationNum = new int *[m_iNumberY]; // iteration step taken at each POI
	double **m_dZNCC = new double *[m_iNumberY]; // ZNCC at each POI
	int **m_iU = new int *[m_iNumberY]; // initial guess u
	int **m_iV = new int *[m_iNumberY]; // initial guess v
	double ***m_dPXY = new double **[m_iNumberY]; //location of each POI in the global coordinate system
	double ***m_dP = new double **[m_iNumberY]; // parameter of deformation p
	double ***m_dDP = new double **[m_iNumberY]; // increment of p
	int **m_iFlag1 = new int *[m_iNumberY]; // flag matrix for all black subset at each POI
	int **m_iFlag2 = new int *[m_iNumberY]; // flag matrix for out of bundary issue when construct the target subset at each POI


	for (i = 0; i < m_iNumberY; i++)
	{
		m_iIterationNum[i] = new int[m_iNumberX];
		m_dZNCC[i] = new double[m_iNumberX];
		m_iU[i] = new int[m_iNumberX];
		m_iV[i] = new int[m_iNumberX];
		m_dP[i] = new double *[m_iNumberX];
		m_dDP[i] = new double *[m_iNumberX];
		m_dPXY[i] = new double *[m_iNumberX];
		m_iFlag1[i] = new int[m_iNumberX];
		m_iFlag2[i] = new int[m_iNumberX];
		for (j = 0; j < m_iNumberX; j++)
		{
			m_iIterationNum[i][j] = 0;
			m_dZNCC[i][j] = 0;
			m_dP[i][j] = new double[6];
			m_dDP[i][j] = new double[6];
			m_dPXY[i][j] = new double[2];
			m_iFlag1[i][j] = 0;
			m_iFlag2[i][j] = 0;

			m_dPXY[i][j][0] = double(m_iMarginX + m_iSubsetY + i * m_iGridSpaceY);
			m_dPXY[i][j][1] = double(m_iMarginY + m_iSubsetX + j * m_iGridSpaceX);
		}
	}

	// Warp matrix
	static double m_dWarp[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
	// Hessian matrix
	static double m_dHessian[6][6] = { { 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0 } };
	// Inverse of the Hessian matrix
	static double m_dInvHessian[6][6] = { { 1, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0 }, { 0, 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 0, 1 } };
	// The item to be divided by the Hessian matrix
	static double m_dNumerator[6] = { 0, 0, 0, 0, 0, 0 };

	double **m_dSubsetR = new double *[m_iSubsetH]; //subset window in R
	double **m_dSubsetT = new double *[m_iSubsetH]; // subset window in T
	double ***m_dRDescent = new double **[m_iSubsetH]; // the steepest descent image DealtR*dW/dp
	double ****m_dHessianXY = new double ***[m_iSubsetH]; // Hessian matrix at each point in subset R
	double m_dSubAveR, m_dSubAveT, m_dSubNorR, m_dSubNorT;
	double **m_dSubsetAveR = new double *[m_iSubsetH]; // (R_i - R_m) / sqrt (Sigma(R_i - R_m)^2)
	double **m_dSubsetAveT = new double *[m_iSubsetH]; // (T_i - T_m) / sqrt (Sigma(T_i - T_m)^2)
	double **m_dError = new double *[m_iSubsetH]; // Error matrix in subset R
	double ****m_dJacobian = new double ***[m_iSubsetH]; // Jacobian matrix dW/dp in subset R

	for (i = 0; i < m_iSubsetH; i++)
	{
		m_dSubsetR[i] = new double[m_iSubsetW];
		m_dSubsetT[i] = new double[m_iSubsetW];
		m_dSubsetAveR[i] = new double[m_iSubsetW];
		m_dSubsetAveT[i] = new double[m_iSubsetW];
		m_dError[i] = new double[m_iSubsetW];
		m_dRDescent[i] = new double *[m_iSubsetW];
		m_dHessianXY[i] = new double **[m_iSubsetW];
		m_dJacobian[i] = new double **[m_iSubsetW];
		for (j = 0; j < m_iSubsetW; j++)
		{
			m_dRDescent[i][j] = new double[6];
			m_dHessianXY[i][j] = new double *[6];
			m_dJacobian[i][j] = new double *[2];
			for (l = 0; l < 6; l++)
			{
				m_dHessianXY[i][j][l] = new double[6];
			}
			for (k = 0; k < 2; k++)
			{
				m_dJacobian[i][j][k] = new double[6];
				for (l = 0; l < 6; l++)
				{
					m_dJacobian[i][j][k][l] = 0;
				}
			}
		}
	}

	// Initialize the data structure for FFTW
	double *m_Subset1 = new double[m_iFFTSubW * m_iFFTSubH]; // subset R
	double *m_Subset2 = new double[m_iFFTSubW * m_iFFTSubH]; // subset T
	double *m_SubsetC = new double[m_iFFTSubW * m_iFFTSubH]; // matrix C

	fftw_complex *m_FreqDom1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)* m_iFFTSubW * (m_iFFTSubH / 2 + 1));
	fftw_complex *m_FreqDom2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)* m_iFFTSubW * (m_iFFTSubH / 2 + 1));
	fftw_complex *m_FreqDomfg = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)* m_iFFTSubW * (m_iFFTSubH / 2 + 1));

	fftw_plan m_fftwPlan1 = fftw_plan_dft_r2c_2d(m_iFFTSubW, m_iFFTSubH, m_Subset1, m_FreqDom1, FFTW_ESTIMATE);
	fftw_plan m_fftwPlan2 = fftw_plan_dft_r2c_2d(m_iFFTSubW, m_iFFTSubH, m_Subset2, m_FreqDom2, FFTW_ESTIMATE);
	fftw_plan m_rfftwPlan = fftw_plan_dft_c2r_2d(m_iFFTSubW, m_iFFTSubH, m_FreqDomfg, m_SubsetC, FFTW_ESTIMATE);

	double m_dAvef, m_dAveg, m_dModf, m_dModg, m_dCorrPeak;
	int m_iCorrPeakXY, m_iCorrPeakX, m_iCorrPeakY;
	double m_dU, m_dUx, m_dUy, m_dV, m_dVx, m_dVy;
	double m_dDU, m_dDUx, m_dDUy, m_dDV, m_dDVx, m_dDVy;
	double m_dWarpX, m_dWarpY;
	int m_iIteration, m_iTemp, m_iTempX, m_iTempY;

	//Start timer for FFT-CC algorithm + IC-GN algorithm
	QueryPerformanceCounter(&m_Begin);
	for (i = 0; i < m_iNumberY; i++)
	{
		for (j = 0; j < m_iNumberX; j++)
		{
			// Initialize the flag parameters
			m_iBlackSubsetFlag = 0;
			m_iOutofBoundaryFlag = 0;

			// Start timer for FFT-CC algorithm
			QueryPerformanceCounter(&m_Start1);
			m_dAvef = 0; // R_m
			m_dAveg = 0; // T_m
			// Feed the gray intensity values into subsets
			for (l = 0; l < m_iFFTSubH; l++)
			{
				for (m = 0; m < m_iFFTSubW; m++)
				{
					m_Subset1[(l * m_iFFTSubW + m)] = m_dR[int(m_dPXY[i][j][0] - m_iSubsetY + l)*m_iWidth+/*][*/int(m_dPXY[i][j][1] - m_iSubsetX + m)];
					//tempr = m_dR[int(m_dPXY[i][j][0] - m_iSubsetY + l)*m_iWidth+/*][*/int(m_dPXY[i][j][1] - m_iSubsetX + m)];
					m_dAvef += (m_Subset1[l * m_iFFTSubW + m] / (m_iFFTSubH * m_iFFTSubW));
					m_Subset2[(l * m_iFFTSubW + m)] = m_dT[int(m_dPXY[i][j][0] - m_iSubsetY + l)*m_iWidth+/*][*/int(m_dPXY[i][j][1] - m_iSubsetX + m)];
					//tempt = m_dT[int(m_dPXY[i][j][0] - m_iSubsetY + l)*m_iWidth+/*][*/int(m_dPXY[i][j][1] - m_iSubsetX + m)];
					m_dAveg += (m_Subset2[l * m_iFFTSubW + m] / (m_iFFTSubH * m_iFFTSubW));
				}
			}
			m_dModf = 0; // sqrt (Sigma(R_i - R_m)^2)
			m_dModg = 0; // sqrt (Sigma(T_i - T_m)^2)
			for (l = 0; l < m_iFFTSubH; l++)
			{
				for (m = 0; m < m_iFFTSubW; m++)
				{
					m_Subset1[(l * m_iFFTSubW + m)] -= m_dAvef;
					m_Subset2[(l * m_iFFTSubW + m)] -= m_dAveg;
					m_dModf += pow((m_Subset1[l * m_iFFTSubW + m]), 2);
					m_dModg += pow((m_Subset2[l * m_iFFTSubW + m]), 2);
				}
			}
			// if one of the two subsets is full of zero intensity, set p = 0, skip this POI
			if (m_dModf == 0 || m_dModg == 0)
			{
				MessageBox(_T("Black subset!"), NULL, MB_OK);
				m_dP[i][j][0] = 0;
				m_dP[i][j][1] = 0;
				m_dP[i][j][2] = 0;
				m_dP[i][j][3] = 0;
				m_dP[i][j][4] = 0;
				m_dP[i][j][5] = 0;
				m_iBlackSubsetFlag = 1; //set flag
				m_iFlag1[i][j] = 1;
				break; //get out of the loop, do not perform any registration algorithm
			}

			//FFT-CC algorithm accelerated by FFTW
			fftw_execute(m_fftwPlan1);
			fftw_execute(m_fftwPlan2);
			for (n = 0; n < m_iFFTSubW * (m_iFFTSubH / 2 + 1); n++)
			{
				m_FreqDomfg[n][0] = (m_FreqDom1[n][0] * m_FreqDom2[n][0]) + (m_FreqDom1[n][1] * m_FreqDom2[n][1]);
				m_FreqDomfg[n][1] = (m_FreqDom1[n][0] * m_FreqDom2[n][1]) - (m_FreqDom1[n][1] * m_FreqDom2[n][0]);
			}
			fftw_execute(m_rfftwPlan);
			m_dCorrPeak = -2; // maximum C
			m_iCorrPeakXY = 0; // loacatoin of maximum C
			m_dTemp = sqrt(m_dModf * m_dModg) * m_iFFTSubW * m_iFFTSubH; //parameter for normalization

			// Search for maximum C, meanwhile normalize C
			for (k = 0; k < m_iFFTSubW * m_iFFTSubH; k++)
			{
				m_SubsetC[k] /= m_dTemp;
				if (m_dCorrPeak < m_SubsetC[k])
				{
					m_dCorrPeak = m_SubsetC[k];
					m_iCorrPeakXY = k;
				}
			}
			// calculate the loacation of maximum C
			m_iCorrPeakX = m_iCorrPeakXY % m_iFFTSubW;
			m_iCorrPeakY = int(m_iCorrPeakXY / m_iFFTSubW);

			// Shift the C peak to the right quadrant 
			if (m_iCorrPeakX > m_iSubsetX)
			{
				m_iCorrPeakX -= m_iFFTSubW;
			}
			if (m_iCorrPeakY > m_iSubsetY)
			{
				m_iCorrPeakY -= m_iFFTSubH;
			}
			m_dZNCC[i][j] = m_dCorrPeak; // save the ZNCC
			// Stop the timer for FFT-CC algorithm and calculate the time consumed
			QueryPerformanceCounter(&m_Stop1);
			m_dFFTTime += 1000 * (m_Stop1.QuadPart - m_Start1.QuadPart) / double(m_Freq.QuadPart); //unit: millisec

			// Stop the timer for IC-GN algorithm
			QueryPerformanceCounter(&m_Start2);
			//Initialize matrix P and DP
			m_iU[i][j] = m_iCorrPeakX; // integer-pixel u
			m_iV[i][j] = m_iCorrPeakY; // integer-pixel v

			// Transfer the initial guess to IC-GN algorithm
			m_dU = m_iCorrPeakX;
			m_dV = m_iCorrPeakY;
			m_dUx = 0;
			m_dUy = 0;
			m_dVx = 0;
			m_dVy = 0;

			m_dP[i][j][0] = m_dU;
			m_dP[i][j][1] = m_dUx;
			m_dP[i][j][2] = m_dUy;
			m_dP[i][j][3] = m_dV;
			m_dP[i][j][4] = m_dVx;
			m_dP[i][j][5] = m_dVy;

			// Initialize the warp matrix
			m_dWarp[0][0] = 1 + m_dUx;
			m_dWarp[0][1] = m_dUy;
			m_dWarp[0][2] = m_dU;
			m_dWarp[1][0] = m_dVx;
			m_dWarp[1][1] = 1 + m_dVy;
			m_dWarp[1][2] = m_dV;
			m_dWarp[2][0] = 0;
			m_dWarp[2][1] = 0;
			m_dWarp[2][2] = 1;

			// Initialize the Hessian matrix in subset R
			for (k = 0; k < 6; k++)
			{
				for (n = 0; n < 6; n++)
				{
					m_dHessian[k][n] = 0;
				}
			}

			// Initialize Subset R
			m_dSubAveR = 0; // R_m
			m_dSubNorR = 0; // T_m
			// Feed the gray intensity to subset R
			for (l = 0; l < m_iSubsetH; l++)
			{
				for (m = 0; m < m_iSubsetW; m++)
				{
					m_dSubsetR[l][m] = m_dR[int(m_dPXY[i][j][0] - m_iSubsetY + l)*m_iWidth+/*][*/int(m_dPXY[i][j][1] - m_iSubsetX + m)];
					m_dSubAveR += (m_dSubsetR[l][m] / (m_iSubsetH * m_iSubsetW));

					// Evaluate the Jacbian dW/dp at (x, 0);
					m_dJacobian[l][m][0][0] = 1;
					m_dJacobian[l][m][0][1] = m - m_iSubsetX;
					m_dJacobian[l][m][0][2] = l - m_iSubsetY;
					m_dJacobian[l][m][0][3] = 0;
					m_dJacobian[l][m][0][4] = 0;
					m_dJacobian[l][m][0][5] = 0;
					m_dJacobian[l][m][1][0] = 0;
					m_dJacobian[l][m][1][1] = 0;
					m_dJacobian[l][m][1][2] = 0;
					m_dJacobian[l][m][1][3] = 1;
					m_dJacobian[l][m][1][4] = m - m_iSubsetX;
					m_dJacobian[l][m][1][5] = l - m_iSubsetY;

					// Compute the steepest descent image DealtR*dW/dp
					for (k = 0; k < 6; k++)
					{
						m_dRDescent[l][m][k] = m_dRx[int(m_dPXY[i][j][0] - m_iSubsetY + l)*m_iWidth+/*][*/int(m_dPXY[i][j][1] - m_iSubsetX + m)] * m_dJacobian[l][m][0][k] + m_dRy[int(m_dPXY[i][j][0] - m_iSubsetY + l)*m_iWidth+/*][*/int(m_dPXY[i][j][1] - m_iSubsetX + m)] * m_dJacobian[l][m][1][k];
					}

					// Compute the Hessian matrix
					for (k = 0; k < 6; k++)
					{
						for (n = 0; n < 6; n++)
						{
							m_dHessianXY[l][m][k][n] = m_dRDescent[l][m][k] * m_dRDescent[l][m][n]; // Hessian matrix at each point
							m_dHessian[k][n] += m_dHessianXY[l][m][k][n]; // sum of Hessian matrix at all the points in subset R
						}
					}
				}
			}

			for (l = 0; l < m_iSubsetH; l++)
			{
				for (m = 0; m < m_iSubsetW; m++)
				{
					m_dSubsetAveR[l][m] = m_dSubsetR[l][m] - m_dSubAveR; // R_i - R_m
					m_dSubNorR += pow(m_dSubsetAveR[l][m], 2);
				}
			}
			m_dSubNorR = sqrt(m_dSubNorR); // sqrt (Sigma(R_i - R_m)^2)

			// Invert the Hessian matrix (Gauss-Jordan algorithm)
			for (l = 0; l < 6; l++)
			{
				for (m = 0; m < 6; m++)
				{
					if (l == m)
					{
						m_dInvHessian[l][m] = 1;
					}
					else
					{
						m_dInvHessian[l][m] = 0;
					}
				}
			}

			for (l = 0; l < 6; l++)
			{
				//Find pivot (maximum lth column element) in the rest (6-l) rows
				m_iTemp = l;
				for (m = l + 1; m < 6; m++)
				{
					if (m_dHessian[m][l] > m_dHessian[m_iTemp][l])
					{
						m_iTemp = m;
					}
				}
				if (fabs(m_dHessian[m_iTemp][l]) == 0)
				{
					MessageBox(_T("Too small element for matrix inverse!"), NULL, MB_OK);
					return;
				}
				// Swap the row which has maximum lth column element
				if (m_iTemp != l)
				{
					for (k = 0; k < 6; k++)
					{
						m_dTemp = m_dHessian[l][k];
						m_dHessian[l][k] = m_dHessian[m_iTemp][k];
						m_dHessian[m_iTemp][k] = m_dTemp;

						m_dTemp = m_dInvHessian[l][k];
						m_dInvHessian[l][k] = m_dInvHessian[m_iTemp][k];
						m_dInvHessian[m_iTemp][k] = m_dTemp;
					}
				}
				// Perform row operation to form required identity matrix out of the Hessian matrix
				for (m = 0; m < 6; m++)
				{
					m_dTemp = m_dHessian[m][l];
					if (m != l)
					{
						for (n = 0; n < 6; n++)
						{
							m_dInvHessian[m][n] -= m_dInvHessian[l][n] * m_dTemp / m_dHessian[l][l];
							m_dHessian[m][n] -= m_dHessian[l][n] * m_dTemp / m_dHessian[l][l];
						}
					}
					else
					{
						for (n = 0; n < 6; n++)
						{
							m_dInvHessian[m][n] /= m_dTemp;
							m_dHessian[m][n] /= m_dTemp;
						}
					}
				}
			}

			// Initialize DeltaP,
			m_dDU = 0;
			m_dDUx = 0;
			m_dDUy = 0;
			m_dDV = 0;
			m_dDVx = 0;
			m_dDVy = 0;

			// Perform interative optimization, with pre-set maximum iteration step
			for (m_iIteration = 0; m_iIteration < m_iMaxIteration; m_iIteration++)
			{
				// Fill warpped image into Subset T
				m_dSubAveT = 0;
				m_dSubNorT = 0;
				for (l = 0; l < m_iSubsetH; l++)
				{
					if (m_iOutofBoundaryFlag != 0)
					{
						break; // if the loacation of the warped subset T is out of the ROI, stop iteration and set p as the current value
					}
					for (m = 0; m < m_iSubsetW; m++)
					{
						// Calculate the location of warped subset T
						m_dWarpX = m_dPXY[i][j][1] + m_dWarp[0][0] * (m - m_iSubsetX) + m_dWarp[0][1] * (l - m_iSubsetY) + m_dWarp[0][2];
						m_dWarpY = m_dPXY[i][j][0] + m_dWarp[1][0] * (m - m_iSubsetX) + m_dWarp[1][1] * (l - m_iSubsetY) + m_dWarp[1][2];
						m_iTempX = int(m_dWarpX);
						m_iTempY = int(m_dWarpY);

						if ((m_iTempX >= 0) && (m_iTempY >= 0) && (m_iTempX < m_iWidth) && (m_iTempY < m_iHeight))
						{
							m_dTempX = m_dWarpX - m_iTempX;
							m_dTempY = m_dWarpY - m_iTempY;
							// if it is integer-pixel location, feed the gray intensity of T into the subset T
							if ((m_dTempX == 0) && (m_dTempY == 0))
							{
								m_dSubsetT[l][m] = m_dT[m_iTempY*m_iWidth+/*][*/m_iTempX];
							}
							else
							{
								// If it is sub-pixel location, estimate the gary intensity using interpolation
								m_dSubsetT[l][m] = 0;
								for (k = 0; k < 4; k++)
								{
									for (n = 0; n < 4; n++)
									{
										m_dSubsetT[l][m] += m_dTBicubic[((m_iTempY*m_iWidth+m_iTempX)*4+k)*4+n]/*[m_iTempY][m_iTempX][k][n]*/ * pow(m_dTempY, k) * pow(m_dTempX, n); 
									}
								}
							}
							m_dSubAveT += (m_dSubsetT[l][m] / (m_iSubsetH * m_iSubsetW));
						}
						else
						{
							m_iOutofBoundaryFlag = 1;
							break;
						}
					}
				}
				if (m_iOutofBoundaryFlag != 0)
				{
					m_iFlag2[i][j] = 1; // save the flag
					break;
				}
				for (l = 0; l < m_iSubsetH; l++)
				{
					for (m = 0; m < m_iSubsetW; m++)
					{
						m_dSubsetAveT[l][m] = m_dSubsetT[l][m] - m_dSubAveT; // T_i - T_m
						m_dSubNorT += pow(m_dSubsetAveT[l][m], 2);
					}
				}
				m_dSubNorT = sqrt(m_dSubNorT); // sqrt (Sigma(T_i - T_m)^2)

				// Compute the error image
				for (k = 0; k < 6; k++)
				{
					m_dNumerator[k] = 0;
				}
				for (l = 0; l < m_iSubsetH; l++)
				{
					for (m = 0; m < m_iSubsetW; m++)
					{
						m_dError[l][m] = (m_dSubNorR / m_dSubNorT) * m_dSubsetAveT[l][m] - m_dSubsetAveR[l][m];

						// Compute the numerator
						for (k = 0; k < 6; k++)
						{
							m_dNumerator[k] += (m_dRDescent[l][m][k] * m_dError[l][m]);
						}
					}
				}

				// Compute DeltaP
				for (k = 0; k < 6; k++)
				{
					m_dDP[i][j][k] = 0;
					for (n = 0; n < 6; n++)
					{
						m_dDP[i][j][k] += (m_dInvHessian[k][n] * m_dNumerator[n]);
					}
				}
				m_dDU = m_dDP[i][j][0];
				m_dDUx = m_dDP[i][j][1];
				m_dDUy = m_dDP[i][j][2];
				m_dDV = m_dDP[i][j][3];
				m_dDVx = m_dDP[i][j][4];
				m_dDVy = m_dDP[i][j][5];

				// Update the warp
				m_dTemp = (1 + m_dDUx) * (1 + m_dDVy) - m_dDUy * m_dDVx;
				// Check if W(Dp) can be inverted
				if (m_dTemp == 0)
				{
					m_strMessage.Format(_T("Non-invertible warp matrix at point (%d, %d)"), j, i);
					MessageBox(m_strMessage, NULL, MB_OK);
					return;
				}

				//W(P) <- W(P) o W(DP)^-1
				m_dWarp[0][0] = ((1 + m_dUx) * (1 + m_dDVy) - m_dUy * m_dDVx) / m_dTemp;
				m_dWarp[0][1] = (m_dUy * (1 + m_dDUx) - (1 + m_dUx) * m_dDUy) / m_dTemp;
				m_dWarp[0][2] = m_dU + (m_dUy * (m_dDU * m_dDVx - m_dDV - m_dDV * m_dDUx) - (1 + m_dUx) * (m_dDU * m_dDVy + m_dDU - m_dDUy * m_dDV)) / m_dTemp;
				m_dWarp[1][0] = (m_dVx * (1 + m_dDVy) - (1 + m_dVy) * m_dDVx) / m_dTemp;
				m_dWarp[1][1] = ((1 + m_dVy) * (1 + m_dDUx) - m_dVx * m_dDUy) / m_dTemp;
				m_dWarp[1][2] = m_dV + ((1 + m_dVy) * (m_dDU * m_dDVx - m_dDV - m_dDV * m_dDUx) - m_dVx * (m_dDU * m_dDVy + m_dDU - m_dDUy * m_dDV)) / m_dTemp;
				m_dWarp[2][0] = 0;
				m_dWarp[2][1] = 0;
				m_dWarp[2][2] = 1;


				// Update DeltaP
				m_dP[i][j][0] = m_dWarp[0][2];
				m_dP[i][j][1] = m_dWarp[0][0] - 1;
				m_dP[i][j][2] = m_dWarp[0][1];
				m_dP[i][j][3] = m_dWarp[1][2];
				m_dP[i][j][4] = m_dWarp[1][0];
				m_dP[i][j][5] = m_dWarp[1][1] - 1;

				m_dU = m_dP[i][j][0];
				m_dUx = m_dP[i][j][1];
				m_dUy = m_dP[i][j][2];
				m_dV = m_dP[i][j][3];
				m_dVx = m_dP[i][j][4];
				m_dVy = m_dP[i][j][5];

				//Check if the norm of DeltaP is small enough
				if (sqrt(pow(m_dDP[i][j][0], 2) + pow(m_dDP[i][j][1] * m_iSubsetX, 2) + pow(m_dDP[i][j][2] * m_iSubsetY, 2) + pow(m_dDP[i][j][3], 2) + pow(m_dDP[i][j][4] * m_iSubsetX, 2) + pow(m_dDP[i][j][5] * m_iSubsetY, 2)) < m_dNormDeltaP)
				{
					break;
				}
			}
			m_iIterationNum[i][j] = m_iIteration; // save iteration steps taken at this POI

			// Stop the timer for IC-GN algorithm and calculate the time consumed
			QueryPerformanceCounter(&m_Stop2);
			m_dICGNTime += 1000 * (m_Stop2.QuadPart - m_Start2.QuadPart) / double(m_Freq.QuadPart); //unit: millisec
		}
	}
	// Stop the timer for FFT-CC algorithm + IC-GN algorithm and calculate the time consumed
	QueryPerformanceCounter(&m_End);
	m_dConsumedTime = 1000 * (m_End.QuadPart - m_Begin.QuadPart) / double(m_Freq.QuadPart) + m_dPrecomputeTime; //unit: millisec

	//Output data as two text files in the diretory of target image
	CString m_sTextPath;
	ofstream m_TextFile;
	m_iIteration = 0;
	m_sTextPath = m_sOutputFilePath + _T("\\\\") + _T("Results_data.txt");
	m_TextFile.open(m_sTextPath, ios::out | ios::trunc);
	// Write detailed data into Results_data.txt
	m_TextFile << "X" << ", " << "Y" << ", " << "Int U" << ", " << "U" << ", " << "Ux" << ", " << "Uy" << ", " << "Int V" << ", " << "V" << ", " << "Vx" << ", " << "Vy" << ", " << "Interation" << ", " << "ZNCC" << ", " << "Black Subset" << ", " << "Out of Boundary" << ", " << endl;
	for (i = 0; i < m_iNumberY; i++)
	{
		for (j = 0; j < m_iNumberX; j++)
		{
			m_iIteration += m_iIterationNum[i][j];
			m_TextFile << int(m_dPXY[i][j][1]) << ", " << int(m_dPXY[i][j][0]) << ", " << m_iU[i][j] << ", " << m_dP[i][j][0] << ", " << m_dP[i][j][1] << ", " << m_dP[i][j][2] << ", " << m_iV[i][j] << ", " << m_dP[i][j][3] << ", " << m_dP[i][j][4] << ", " << m_dP[i][j][5] << ", " << m_iIterationNum[i][j] << ", " << m_dZNCC[i][j] << ", " << m_iFlag1[i][j] << ", " << m_iFlag2[i][j] << endl;
		}
	}
	m_TextFile.close();

	//Output information of stastics into Results_info.txt
	m_sTextPath = m_sOutputFilePath + _T("\\\\") + _T("Results_info.txt");
	m_TextFile.open(m_sTextPath, ios::out | ios::trunc);

	m_TextFile << "Interval (X-axis): " << m_iGridSpaceX << " [pixel]" << endl;
	m_TextFile << "Interval (Y-axis): " << m_iGridSpaceY << " [pixel]" << endl;
	m_TextFile << "Number of POI: " << m_iNumberY*m_iNumberX << " = " << m_iNumberX << " X " << m_iNumberY << endl;
	m_TextFile << "Subset dimension: " << m_iSubsetW << "x" << m_iSubsetH << " pixels" << endl;
	m_TextFile << "Time comsumed: " << m_dConsumedTime << " [millisec]" << endl;
	m_TextFile << "Time for Pre-computation: " << m_dPrecomputeTime << " [millisec]" << endl;
	m_TextFile << "Time for integral-pixel registration: " << m_dFFTTime / (m_iNumberY*m_iNumberX) << " [millisec]" << endl;
	m_TextFile << "Time for sub-pixel registration: " << m_dICGNTime / (m_iNumberY*m_iNumberX) << " [millisec]" << " for average iteration steps of " << double(m_iIteration) / (m_iNumberY*m_iNumberX) << endl;
	m_TextFile << m_iWidth << ", " << m_iHeight << ", " << m_iGridSpaceX << ", " << m_iGridSpaceY << ", " << endl;

	m_TextFile.close();

	//Destroy the created matrices
	for (i = 0; i < m_iSubsetH; i++)
	{
		for (j = 0; j < m_iSubsetW; j++)
		{
			for (l = 0; l < 6; l++)
			{
				delete[]m_dHessianXY[i][j][l];
			}
			for (k = 0; k < 2; k++)
			{
				delete[]m_dJacobian[i][j][k];
			}
			delete[]m_dRDescent[i][j];
			delete[]m_dHessianXY[i][j];
			delete[]m_dJacobian[i][j];
		}
		delete[]m_dSubsetR[i];
		delete[]m_dSubsetT[i];
		delete[]m_dSubsetAveR[i];
		delete[]m_dSubsetAveT[i];
		delete[]m_dError[i];
		delete[]m_dRDescent[i];
		delete[]m_dHessianXY[i];
		delete[]m_dJacobian[i];
	}

	delete[]m_dSubsetR;
	delete[]m_dSubsetT;
	delete[]m_dSubsetAveR;
	delete[]m_dSubsetAveT;
	delete[]m_dError;
	delete[]m_dRDescent;
	delete[]m_dHessianXY;
	delete[]m_dJacobian;

	fftw_destroy_plan(m_fftwPlan1);
	fftw_destroy_plan(m_fftwPlan2);
	fftw_destroy_plan(m_rfftwPlan);
	fftw_free(m_FreqDom1);
	fftw_free(m_FreqDom2);
	fftw_free(m_FreqDomfg);

	delete[]m_Subset1;
	delete[]m_Subset2;
	delete[]m_SubsetC;

	for (i = 0; i < m_iNumberY; i++)
	{
		for (j = 0; j < m_iNumberX; j++)
		{
			delete[]m_dP[i][j];
			delete[]m_dDP[i][j];
			delete[]m_dPXY[i][j];

		}
		delete[]m_iU[i];
		delete[]m_iV[i];
		delete[]m_dP[i];
		delete[]m_dDP[i];
		delete[]m_dPXY[i];
		delete[]m_iFlag1[i];
		delete[]m_iFlag2[i];
		delete[]m_dZNCC[i];
		delete[]m_iIterationNum[i];
	}
	delete[]m_iU;
	delete[]m_iV;
	delete[]m_dP;
	delete[]m_dDP;
	delete[]m_dPXY;
	delete[]m_iFlag1;
	delete[]m_iFlag2;
	delete[]m_dZNCC;
	delete[]m_iIterationNum;

	free(m_dT);
	free(m_dR);
	free(m_dRx);
	free(m_dRy);
	free(m_dTx);
	free(m_dTy);
	free(m_dTxy);
	/*delete[]m_dTaoT;
	delete[]m_dAlphaT;

	for (i; i < m_iHeight; i++)
	{
		delete[]m_dR[i];
		delete[]m_dRx[i];
		delete[]m_dRy[i];

		delete[]m_dT[i];
		delete[]m_dTx[i];
		delete[]m_dTy[i];
		delete[]m_dTxy[i];
	}*/

	/*delete[]m_dT;
	delete[]m_dTx;
	delete[]m_dTy;
	delete[]m_dTxy;

	delete[]m_dR;
	delete[]m_dRx;
	delete[]m_dRy;*/
	free(m_dTBicubic);
	//for (i; i < m_iHeight; i++)
	//{
	//	for (j; j < m_iWidth; j++)
	//	{
	//		for (k; k < 4; k++)
	//		{
	//			delete[]m_dTBicubic[i][j][k];
	//		}
	//		delete[]m_dTBicubic[i][j];
	//	}
	//	delete[]m_dTBicubic[i];
	//}
	//delete[]m_dTBicubic;
	free(m_dImg1);
	free(m_dImg2);

	//for (i = 0; i < m_iImgHeight; i++)
	//{
	//	delete[]m_dImg1[i];
	//	delete[]m_dImg2[i];
	//}
	//delete[]m_dImg1;
	//delete[]m_dImg2;

	// Destroy CImage objects
	m_Image1.Destroy();
	m_Image2.Destroy();

	// Pop up a dialog for completion
	m_strMessage.Format(_T("Path-independent DIC (FFT-CC + IC-GN algorithm) took %f milliseconds"), m_dConsumedTime);
	MessageBox(m_strMessage, NULL, MB_OK);
}



