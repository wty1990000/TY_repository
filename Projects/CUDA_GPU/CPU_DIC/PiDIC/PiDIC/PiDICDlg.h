
// PiDICDlg.h : header file
//

#pragma once


// CPiDICDlg dialog
class CPiDICDlg : public CDialogEx
{
// Construction
public:
	CPiDICDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	enum { IDD = IDD_PIDIC_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedSelectrefbtn();
	CImage m_Image1, m_Image2;
	CString m_sFilePath, m_sFileName, m_sFileExt;
	CString m_sOutputFilePath;
	CString m_sRefImgPath;
	CString m_sTarImgPath;
	int m_iMarginY;
	int m_iMarginX;
	int m_iGridSpaceX;
	int m_iGridSpaceY;
	int m_iSubsetX;
	int m_iSubsetY;
	double m_dNormDeltaP;
	int m_iMaxIteration;
	afx_msg void OnBnClickedOk();
	afx_msg void OnBnClickedSelecttarbtn();

};
