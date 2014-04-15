
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
	m_dNormDeltaP = 0.001f;
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

	CString m_strMessage; 	//message for pop-out dialog
	//Read images and convert the gray value of intensity to float precision number
	/*----------------Linearlize---------------------*/
	float *m_dImg1 = (float*)malloc(m_iImgHeight*m_iImgWidth*sizeof(float));
	float *m_dImg2 = (float*)malloc(m_iImgHeight*m_iImgWidth*sizeof(float));

	float m_dTemp;
	COLORREF m_PixelColor;

	for (int i = 0; i < m_iImgHeight; i++){
		for (int j = 0; j < m_iImgWidth; j++){
			m_PixelColor = m_Image1.GetPixel(j, i);
			m_dTemp = float((GetRValue(m_PixelColor) + GetGValue(m_PixelColor) + GetBValue(m_PixelColor)) / 3);
			m_dImg1[i*m_iImgWidth+j] = m_dTemp;
			m_PixelColor = m_Image2.GetPixel(j, i);
			m_dTemp = float((GetRValue(m_PixelColor) + GetGValue(m_PixelColor) + GetBValue(m_PixelColor)) / 3);
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
	//Define the size of subset window for FFT-CC algorithm
	int m_iFFTSubW = m_iSubsetX * 2;
	int m_iFFTSubH = m_iSubsetY * 2;
	//Estimate the number of points of interest(POIs)
	int m_iNumberX = int(floor((m_iWidth - m_iSubsetX * 2 - m_iMarginX * 2) / float(m_iGridSpaceX))) + 1;
	int m_iNumberY = int(floor((m_iHeight - m_iSubsetY * 2 - m_iMarginY * 2) / float(m_iGridSpaceY))) + 1;
	//Timer
	float precompute_time, fft_time, icgn_time, total_time;
	precompute_time = fft_time = icgn_time = total_time=0.0;

	//----------------For FFT-CC Inputs/outputs--------------------
	float *m_dZNCC = (float*)malloc(m_iNumberX*m_iNumberY*sizeof(float)); // ZNCC at each POI

	//----------------For ICGN outputs----------------------
	int *m_iU = (int*)malloc(m_iNumberX*m_iNumberY*sizeof(int)); // initial guess u
	int *m_iV = (int*)malloc(m_iNumberX*m_iNumberY*sizeof(int)); // initial guess v
	float *m_dPXY = (float*)malloc(m_iNumberX*m_iNumberY*2*sizeof(float)); //location of each POI in the global coordinate system
	float *m_dP = (float*)malloc(m_iNumberX*m_iNumberY*6*sizeof(float)); // parameter of deformation p	
	int *m_IterationNum = (int*)malloc(m_iNumberX*m_iNumberY*sizeof(int));

	//Fill int the values of m_dPXY
	for (int i = 0; i < m_iNumberY; i++)
	{
		for (int j = 0; j < m_iNumberX; j++)
		{
			m_dPXY[(i*m_iNumberX+j)*2+0] = float(m_iMarginX + m_iSubsetY + i * m_iGridSpaceY);
			m_dPXY[(i*m_iNumberX+j)*2+1] = float(m_iMarginY + m_iSubsetX + j * m_iGridSpaceX);
		}
	}

	combined_functions(m_dImg1, m_dImg2, m_dPXY, m_iWidth, m_iHeight, m_iSubsetH, m_iSubsetW, m_dNormDeltaP,
						m_iSubsetX, m_iSubsetY, m_iNumberX, m_iNumberY, m_iFFTSubH, m_iFFTSubW, m_iMaxIteration,
						m_iU, m_iV,  m_dZNCC, m_dP,m_IterationNum,
						precompute_time, fft_time, icgn_time, total_time);
	
	//Output data as two text files in the diretory of target image
	CString m_sTextPath;
	ofstream m_TextFile;
	m_sTextPath = m_sOutputFilePath + _T("\\\\") + _T("Results_data.txt");
	m_TextFile.open(m_sTextPath, ios::out | ios::trunc);
	// Write detailed data into Results_data.txt
	m_TextFile << "X" << ", " << "Y" << ", " << "Int U" << ", " << "U" << ", " << "Ux" << ", " << "Uy" << ", " << "Int V" << ", " << "V" << ", " << "Vx" << ", " << "Vy" << ", " << "Interation" << ", " << "ZNCC" << ", " << endl;
	for (int i = 0; i < m_iNumberY; i++)
	{
		for (int j = 0; j < m_iNumberX; j++)
		{
			m_TextFile << int(m_dPXY[(i*m_iNumberX+j)*2+1]) << ", " << int(m_dPXY[(i*m_iNumberX+j)*2+0]) << ", " << m_iU[i*m_iNumberX+j] << ", " << m_dP[(i*m_iNumberX+j)*6+0] << ", " << m_dP[(i*m_iNumberX+j)*6+1] << ", " << m_dP[(i*m_iNumberX+j)*6+2] << ", " << m_iV[i*m_iNumberX+j] << ", " << m_dP[(i*m_iNumberX+j)*6+3] << ", " << m_dP[(i*m_iNumberX+j)*6+4] << ", " << m_dP[(i*m_iNumberX+j)*6+5] << ", " << m_IterationNum[i*m_iNumberX+j] << ", " << m_dZNCC[i*m_iNumberX+j] << ", " <<endl;
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
	m_TextFile << "Time comsumed: " << total_time << " [millisec]" << endl;
	m_TextFile << "Time for Pre-computation: " << precompute_time << " [millisec]" << endl;
	m_TextFile << "Time for integral-pixel registration: " << fft_time / (m_iNumberY*m_iNumberX) << " [millisec]" << endl;
	m_TextFile << "Time for sub-pixel registration: " << icgn_time / (m_iNumberY*m_iNumberX) << " [millisec]" << " for iteration steps of " << m_iMaxIteration << endl;
	m_TextFile << m_iWidth << ", " << m_iHeight << ", " << m_iGridSpaceX << ", " << m_iGridSpaceY << ", " << endl;

	m_TextFile.close();

	//Free parameters
	free(m_iU);
	free(m_iV);
	free(m_dP);
	free(m_IterationNum);
	free(m_dPXY);
	free(m_dImg1);
	free(m_dImg2);

	m_Image1.Destroy();
	m_Image2.Destroy();

	// Pop up a dialog for completion
	m_strMessage.Format(_T("Path-independent DIC (FFT-CC + IC-GN algorithm) took %f milliseconds"), total_time);
	MessageBox(m_strMessage, NULL, MB_OK);
}



