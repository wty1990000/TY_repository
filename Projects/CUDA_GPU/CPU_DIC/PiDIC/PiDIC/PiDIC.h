
// PiDIC.h : main header file for the PROJECT_NAME application
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols


// CPiDICApp:
// See PiDIC.cpp for the implementation of this class
//

class CPiDICApp : public CWinApp
{
public:
	CPiDICApp();

// Overrides
public:
	virtual BOOL InitInstance();

// Implementation

	DECLARE_MESSAGE_MAP()
};

extern CPiDICApp theApp;