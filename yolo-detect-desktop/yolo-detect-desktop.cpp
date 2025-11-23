// yolo-detect-desktop.cpp : 定义应用程序的入口点。
//

#include "framework.h"
#include "yolo-detect-desktop.h"

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <array>
#include <vector>
#include <string>
#include <format>
#include <ranges>
#include <limits>
#include <algorithm>
#include <iostream>
#include <commdlg.h>

#define MAX_LOADSTRING 100

#ifndef IDM_OPEN
#define IDM_OPEN  32771 // 定义“打开图片”菜单命令 ID
#endif

// 全局变量:
HINSTANCE hInst;                                // 当前实例
WCHAR szTitle[MAX_LOADSTRING];                  // 标题栏文本
WCHAR szWindowClass[MAX_LOADSTRING];            // 主窗口类名

// 原始图像与检测结果(不再存储已标注的位图，以便高质量缩放重新绘制)
static cv::Mat g_img;               // 原始图像 (BGR)
struct Detection { float x1,y1,x2,y2,conf; int cls; };
static std::vector<Detection> g_detections; // 检测框列表

// 显示相关
static double g_zoom = 1.0;         // 缩放比
static int g_scrollX = 0;           // 滚动偏移(缩放后坐标系)
static int g_scrollY = 0;
static int g_imgWidth = 0;
static int g_imgHeight = 0;
static HBITMAP g_hBaseBmp = nullptr; // 原始图像 HBITMAP (用于 StretchBlt)

static std::wstring g_modelPath = L"C:/Users/33554/Documents/learn/zixue/cpp23/ConsoleApplication1/x64/Release/yolo11x.onnx";

// ====== YOLO 类别名称 ======
static const std::array<std::string, 80> kClassNames = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

// 前向声明
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);
static void RunYoloDetection(const std::string& image_path);
static std::string OpenImageFileDialog(HWND hWnd);
static void UpdateScrollBars(HWND hWnd);
static void ClampScroll();

// ====== 工具: 排序 (按置信度) ======
static void sortVecs(std::vector<std::vector<float>>& preds, size_t confIdx) {
    auto comp = [confIdx](const std::vector<float>& a, const std::vector<float>& b) {
        float va = (a.size() > confIdx) ? a[confIdx] : -std::numeric_limits<float>::infinity();
        float vb = (b.size() > confIdx) ? b[confIdx] : -std::numeric_limits<float>::infinity();
        bool nan_a = std::isnan(va);
        bool nan_b = std::isnan(vb);
        if (nan_a && nan_b) return false;
        if (nan_a) return false;
        if (nan_b) return true;
        return va > vb; // 降序
    };
    std::ranges::sort(preds, comp);
}

// ====== 工具: 非极大值抑制 ======
static std::vector<std::vector<float>> nms(std::vector<std::vector<float>>& preds, float iou_thresh = 0.5f) {
    if (preds.empty()) return {};
    std::vector<std::vector<float>> ret;
    sortVecs(preds, 4);
    while (!preds.empty()) {
        auto best = preds.front();
        ret.push_back(best);
        preds.erase(preds.begin());
        auto it = preds.begin();
        while (it != preds.end()) {
            float x1 = std::max(best[0], (*it)[0]);
            float y1 = std::max(best[1], (*it)[1]);
            float x2 = std::min(best[2], (*it)[2]);
            float y2 = std::min(best[3], (*it)[3]);
            float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float box1_area = (best[2] - best[0]) * (best[3] - best[1]);
            float box2_area = ((*it)[2] - (*it)[0]) * ((*it)[3] - (*it)[1]);
            float union_area = box1_area + box2_area - inter_area;
            float iou = (union_area > 0) ? (inter_area / union_area) : 0.0f;
            if (iou > iou_thresh) {
                it = preds.erase(it);
            } else {
                ++it;
            }
        }
    }
    return ret;
}

// ====== 将 cv::Mat 转为 HBITMAP 以便 GDI 绘制 ======
static HBITMAP MatToHBITMAP(const cv::Mat& mat) {
    if (mat.empty()) return nullptr;
    cv::Mat bgr;
    if (mat.channels() == 3) {
        bgr = mat.clone(); // 我们的可视化目前是 BGR->RGB 之后绘制的，转回 BGR 以符合常见顺序
    } else if (mat.channels() == 4) {
        cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
    } else {
        cv::cvtColor(mat, bgr, cv::COLOR_GRAY2BGR);
    }

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = bgr.cols;
    bmi.bmiHeader.biHeight = -bgr.rows; // 负数表示自顶向下
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 24;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* bits = nullptr;
    HDC hdc = GetDC(nullptr);
    HBITMAP hBitmap = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (hBitmap && bits) {
        size_t lineSize = (size_t)bgr.cols * 3;
        for (int y = 0; y < bgr.rows; ++y) {
            memcpy(static_cast<unsigned char*>(bits) + y * lineSize, bgr.ptr(y), lineSize);
        }
    }
    ReleaseDC(nullptr, hdc);
    return hBitmap;
}

// 文件打开对话框
static std::string OpenImageFileDialog(HWND hWnd) {
	char fileName[MAX_PATH] = { 0 };
	OPENFILENAMEA ofn{};
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = hWnd;
	ofn.lpstrFile = fileName;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrFilter = "Image Files\0*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tif;*.tiff\0All Files\0*.*\0";
	ofn.nFilterIndex = 1;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_EXPLORER;
	if (GetOpenFileNameA(&ofn)) return std::string(fileName);
	return {};
}

// ====== YOLO 推理入口 ======
static void RunYoloDetection(const std::string& image_path) {
	if (image_path.empty()) return;
	int input_w = 640;
	int input_h = 640;
	const float conf_thresh = 0.4f;

	g_detections.clear();
	g_img.release();

	cv::Mat img = cv::imread(image_path);
	if (img.empty()) {
		std::cerr << "无法读取图片: " << image_path << std::endl;
		return;
	}
	g_img = img.clone();
	g_imgWidth = g_img.cols;
	g_imgHeight = g_img.rows;

	try {
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_infer");
		Ort::SessionOptions session_options;
		session_options.SetIntraOpNumThreads(1);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

		OrtCUDAProviderOptions cuda_options;
		cuda_options.device_id = 0;
		session_options.AppendExecutionProvider_CUDA(cuda_options);

		Ort::Session session(env, g_modelPath.c_str(), session_options);
		Ort::AllocatorWithDefaultOptions allocator;

		if (session.GetInputCount() > 0) {
			auto in_tensor_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
			auto in_shape = in_tensor_info.GetShape();
			if (in_shape.size() >= 4) {
				if (in_shape[2] > 0) input_h = (int)in_shape[2];
				if (in_shape[3] > 0) input_w = (int)in_shape[3];
			}
		}
		cv::Mat resized;
		cv::resize(img, resized, cv::Size(input_w, input_h));
		cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
		resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

		std::vector<int64_t> input_shape = { 1,3,input_h,input_w };
		std::vector<float> input_tensor_values(1 * 3 * input_h * input_w);

        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < input_h; ++y) {
                for (int x = 0; x < input_w; ++x) {
                    input_tensor_values[c * input_h * input_w + y * input_w + x] = resized.at<cv::Vec3f>(y, x)[c];
                }
            }
        }

		std::vector<const char*> input_names;
		for (size_t i = 0; i < session.GetInputCount(); ++i) {
			input_names.push_back(session.GetInputNameAllocated(i, allocator).get());
		}

		std::vector<const char*> output_names;
        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            output_names.push_back(session.GetOutputNameAllocated(i, allocator).get());
        }

		Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

		auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());
		if (output_tensors.empty()) {
			std::cerr << "模型没有返回输出" << std::endl;
			return;
		}
		float* out_data = output_tensors[0].GetTensorMutableData<float>();
		auto out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
		if (out_shape.size() < 3) {
			std::cerr << "输出形状不符合预期" << std::endl;
			return;
		}
		int attrs = (int)out_shape[1];
		int num_points = (int)out_shape[2];
		std::vector<std::vector<float>> preds;

		for (int p = 0; p < num_points; ++p) {
			std::vector<float> pred;
			float cx = out_data[0 * num_points + p];
			float cy = out_data[1 * num_points + p];
			float w = out_data[2 * num_points + p];
			float h = out_data[3 * num_points + p];
			int best_cls = -1;
			float best_prob = 0.f;
			for (int c = 4; c < attrs; ++c) {
				float cls_prob = out_data[c * num_points + p];
				if (cls_prob > best_prob) {
					best_prob = cls_prob;
					best_cls = c - 4;
				}
			} 
            if (best_prob < conf_thresh) continue;
			float x1 = (cx - w / 2.f) / input_w * g_imgWidth;
			float y1 = (cy - h / 2.f) / input_h * g_imgHeight;
			float x2 = (cx + w / 2.f) / input_w * g_imgWidth;
			float y2 = (cy + h / 2.f) / input_h * g_imgHeight;
			pred.push_back(std::clamp(x1, 0.f, (float)g_imgWidth - 1));
			pred.push_back(std::clamp(y1, 0.f, (float)g_imgHeight - 1));
			pred.push_back(std::clamp(x2, 0.f, (float)g_imgWidth - 1));
			pred.push_back(std::clamp(y2, 0.f, (float)g_imgHeight - 1));
			pred.push_back(best_prob);
			pred.push_back((float)best_cls);
			preds.push_back(pred);
		}
		preds = nms(preds, 0.5f);
		for (auto& pr : preds) {
			Detection d{ pr[0],pr[1],pr[2],pr[3],pr[4],(int)pr[5] };
			g_detections.push_back(d);
		}
		if (g_hBaseBmp) {
			DeleteObject(g_hBaseBmp);
			g_hBaseBmp = nullptr;
		}
		g_hBaseBmp = MatToHBITMAP(g_img);
		g_zoom = 1.0;
		g_scrollX = g_scrollY = 0;

	}
	catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime 错误: " << e.what() << std::endl;
    }
	catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
    }
	catch (...) {
        std::cerr << "未知错误" << std::endl;
    }
}

static void ClampScroll(){ int scaledW = (int)(g_imgWidth * g_zoom); int scaledH = (int)(g_imgHeight * g_zoom); RECT rc; HWND hWnd = GetActiveWindow(); GetClientRect(hWnd,&rc); int clientW = rc.right - rc.left; int clientH = rc.bottom - rc.top; if(g_scrollX < 0) g_scrollX = 0; if(g_scrollY < 0) g_scrollY = 0; if(g_scrollX > scaledW - clientW) g_scrollX = std::max(0, scaledW - clientW); if(g_scrollY > scaledH - clientH) g_scrollY = std::max(0, scaledH - clientH); }

static void UpdateScrollBars(HWND hWnd){ if(g_img.empty()) { ShowScrollBar(hWnd, SB_BOTH, FALSE); return; } RECT rc; GetClientRect(hWnd,&rc); int clientW = rc.right - rc.left; int clientH = rc.bottom - rc.top; int scaledW = (int)(g_imgWidth * g_zoom); int scaledH = (int)(g_imgHeight * g_zoom); SCROLLINFO si{}; si.cbSize = sizeof(si); si.fMask = SIF_RANGE | SIF_PAGE | SIF_POS; si.nMin = 0; si.nMax = scaledW - 1; si.nPage = clientW; si.nPos = g_scrollX; SetScrollInfo(hWnd, SB_HORZ, &si, TRUE); si.nMax = scaledH - 1; si.nPage = clientH; si.nPos = g_scrollY; SetScrollInfo(hWnd, SB_VERT, &si, TRUE); ShowScrollBar(hWnd, SB_HORZ, scaledW > clientW); ShowScrollBar(hWnd, SB_VERT, scaledH > clientH); }

// ====== Windows 程序入口 ======
int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_YOLODETECTDESKTOP, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    if (!InitInstance (hInstance, nCmdShow)) {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_YOLODETECTDESKTOP));

    MSG msg;

    while (GetMessage(&msg, nullptr, 0, 0)) {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int) msg.wParam;
}

ATOM MyRegisterClass(HINSTANCE hInstance) {
    WNDCLASSEXW wcex;
    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = WndProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_YOLODETECTDESKTOP));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_YOLODETECTDESKTOP);
    wcex.lpszClassName  = szWindowClass;
    wcex.hIconSm        = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));
    return RegisterClassExW(&wcex);
}

BOOL InitInstance(HINSTANCE hInstance, int nCmdShow) {
    hInst = hInstance;

    // 窗口样式（保留滚动条）
    DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_HSCROLL | WS_VSCROLL;

    // 希望的客户区尺寸（像素）
    const int desiredClientW = 1280;
    const int desiredClientH = 720;

    // 计算包含非客户区后的窗口总大小
    RECT rc = { 0, 0, desiredClientW, desiredClientH };
    AdjustWindowRect(&rc, style, TRUE);
    int winW = rc.right - rc.left;
    int winH = rc.bottom - rc.top;

    // ----- 方案 A：居中显示 -----
    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);
    int posX = (screenW - winW) / 2;
    int posY = (screenH - winH) / 2;

    HWND hWnd = CreateWindowW(
        szWindowClass,
        szTitle,
        style,
        posX, posY,
        winW, winH,
        nullptr,
        nullptr,
        hInstance,
        nullptr
    );

    if (!hWnd) return FALSE;

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    return TRUE;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_COMMAND: {
        int wmId = LOWORD(wParam);
        switch (wmId) {
        case IDM_OPEN: {
            auto path = OpenImageFileDialog(hWnd);
            if(!path.empty()){
                RunYoloDetection(path);
                UpdateScrollBars(hWnd);
                InvalidateRect(hWnd,nullptr,TRUE);
            }
        } break;
        case IDM_ABOUT:
            DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
            break;
        case IDM_EXIT:
            DestroyWindow(hWnd);
            break;
        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
        }
    } break;
    case WM_LBUTTONDBLCLK: {
        auto path = OpenImageFileDialog(hWnd);
        if(!path.empty()){
            RunYoloDetection(path);
            UpdateScrollBars(hWnd);
            InvalidateRect(hWnd,nullptr,TRUE);
        }
    } break;
    case WM_SIZE: {
        UpdateScrollBars(hWnd);
    } break;
    case WM_MOUSEWHEEL: {
        if (GET_KEYSTATE_WPARAM(wParam) & MK_CONTROL) {
            int delta = GET_WHEEL_DELTA_WPARAM(wParam);
            POINT pt; GetCursorPos(&pt); ScreenToClient(hWnd,&pt);
            double oldZoom = g_zoom;
            if(delta>0) g_zoom *= 1.1;
            else g_zoom /= 1.1;
            g_zoom = std::clamp(g_zoom, 0.1, 8.0); // 保持鼠标点位置稳定

            g_scrollX = (int)((g_scrollX + pt.x) * (g_zoom/oldZoom) - pt.x);
            g_scrollY = (int)((g_scrollY + pt.y) * (g_zoom/oldZoom) - pt.y);
            ClampScroll();
            UpdateScrollBars(hWnd);
            InvalidateRect(hWnd,nullptr,TRUE);
        } else {
            // 普通滚轮垂直滚动
            int lines = GET_WHEEL_DELTA_WPARAM(wParam)/WHEEL_DELTA * 60;
            g_scrollY -= lines;
            ClampScroll();
            SetScrollPos(hWnd, SB_VERT, g_scrollY, TRUE);
            InvalidateRect(hWnd,nullptr,FALSE);
        }
    } break;
    case WM_HSCROLL: {
        SCROLLINFO si{}; si.cbSize=sizeof(si); si.fMask=SIF_ALL;
        GetScrollInfo(hWnd,SB_HORZ,&si);
        int pos = si.nPos;
        switch(LOWORD(wParam)) {
        case SB_LINELEFT: pos -= 20; break;
        case SB_LINERIGHT: pos += 20; break;
        case SB_PAGELEFT: pos -= (int)si.nPage; break;
        case SB_PAGERIGHT: pos += (int)si.nPage; break;
        case SB_THUMBTRACK: pos = HIWORD(wParam); break;
        }
        g_scrollX = pos;
        ClampScroll();
        SetScrollPos(hWnd, SB_HORZ, g_scrollX, TRUE);
        InvalidateRect(hWnd,nullptr,FALSE);
    } break;
    case WM_VSCROLL: {
        SCROLLINFO si{}; si.cbSize=sizeof(si); si.fMask=SIF_ALL;
        GetScrollInfo(hWnd,SB_VERT,&si);
        int pos = si.nPos;
        switch(LOWORD(wParam)) {
        case SB_LINEUP: pos -= 20; break;
        case SB_LINEDOWN: pos += 20; break;
        case SB_PAGEUP: pos -= (int)si.nPage; break;
        case SB_PAGEDOWN: pos += (int)si.nPage; break;
        case SB_THUMBTRACK: pos = HIWORD(wParam); break;
        }
        g_scrollY = pos;
        ClampScroll();
        SetScrollPos(hWnd, SB_VERT, g_scrollY, TRUE);
        InvalidateRect(hWnd,nullptr,FALSE);
    } break;
    case WM_PAINT: {
        PAINTSTRUCT ps; HDC hdc = BeginPaint(hWnd,&ps);
        RECT rcClient; GetClientRect(hWnd,&rcClient);
        int clientW = rcClient.right - rcClient.left;
        int clientH = rcClient.bottom - rcClient.top;
        SetStretchBltMode(hdc, HALFTONE);

        if(g_hBaseBmp && !g_img.empty()){
            HDC memDC = CreateCompatibleDC(hdc);
            HGDIOBJ old = SelectObject(memDC, g_hBaseBmp);
            int srcX = (int)(g_scrollX / g_zoom);
            int srcY = (int)(g_scrollY / g_zoom);
            int srcW = (int)(clientW / g_zoom);
            int srcH = (int)(clientH / g_zoom); // 边界裁剪

            if(srcX + srcW > g_imgWidth) srcW = g_imgWidth - srcX;
            if(srcY + srcH > g_imgHeight) srcH = g_imgHeight - srcY;

            StretchBlt(hdc, 0,0, clientW, clientH, memDC, srcX, srcY, srcW, srcH, SRCCOPY);
            SelectObject(memDC, old);
            DeleteDC(memDC);

            // 绘制检测框与标签（按缩放重新计算）
            if(!g_detections.empty()){
                double z = g_zoom;
                HFONT hFont = NULL;
                int fontPx = (int)(12 * z);
                if(fontPx<8) fontPx=8;
                LOGFONTW lf{}; lf.lfHeight = -fontPx; lstrcpyW(lf.lfFaceName, L"Segoe UI"); hFont = CreateFontIndirectW(&lf);
                HGDIOBJ oldFont = SelectObject(hdc, hFont);
                for(auto &d: g_detections){
                    // 判断是否在当前视口
                    double x1s = d.x1 * z - g_scrollX;
                    double y1s = d.y1 * z - g_scrollY;
                    double x2s = d.x2 * z - g_scrollX;
                    double y2s = d.y2 * z - g_scrollY;
                    if(x2s<0 || y2s<0 || x1s>clientW || y1s>clientH) continue;
                    RECT box{ (LONG)std::lround(x1s), (LONG)std::lround(y1s), (LONG)std::lround(x2s), (LONG)std::lround(y2s)}; // 框线宽度随缩放
                    HPEN pen = CreatePen(PS_SOLID, (int)std::clamp(z,1.0,5.0), RGB(255,0,0));
                    HGDIOBJ oldPen = SelectObject(hdc, pen);
                    HGDIOBJ oldBrush = SelectObject(hdc, GetStockObject(HOLLOW_BRUSH));
                    Rectangle(hdc, box.left, box.top, box.right, box.bottom);
                    SelectObject(hdc, oldBrush);
                    SelectObject(hdc, oldPen);
                    DeleteObject(pen);

                    std::string label = std::format("{} {:.2f}", kClassNames[d.cls], d.conf);
                    std::wstring wlabel(label.begin(), label.end());
                    SIZE sz;
                    GetTextExtentPoint32W(hdc, wlabel.c_str(), (int)wlabel.size(), &sz);
                    int pad = 2;
                    RECT bg{box.left, box.top - sz.cy - pad*2, box.left + sz.cx + pad*2, box.top};
                    if(bg.top < 0){
                        bg.top = box.top;
                        bg.bottom = box.top + sz.cy + pad*2;
                    }
                    HBRUSH hbr = CreateSolidBrush(RGB(255,255,0));
                    FillRect(hdc,&bg,hbr);
                    DeleteObject(hbr);
                    SetBkMode(hdc, TRANSPARENT);
                    TextOutW(hdc, bg.left + pad, bg.top + pad, wlabel.c_str(), (int)wlabel.size());
                }
                SelectObject(hdc, oldFont);
                DeleteObject(hFont);
            }
        } else {
            // 修复 C6385: TextOutW 的字符数应为字符串实际长度
            TextOutW(hdc, 10, 10, L"选择图片", 4);
        }
        EndPaint(hWnd,&ps);
    } break;
    case WM_DESTROY:
        if(g_hBaseBmp){ DeleteObject(g_hBaseBmp); g_hBaseBmp=nullptr; }
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam) {
    UNREFERENCED_PARAMETER(lParam);
    switch (message) {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;
    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL) {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}
