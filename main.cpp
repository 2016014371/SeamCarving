#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cstdio>
#include <iostream>

#define MAX_COLS 2000
#define INF 1e5

struct MouseArgs // 用于和回调函数交互
{
    cv::Mat& img; // 显示在窗口中的那张图
    cv::Mat& mask; // 用户绘制的选区（删除/保留）
    cv::Vec3b& color; // 用来高亮选区的颜色
    MouseArgs(cv::Mat& img, cv::Mat& mask, cv::Vec3b& color): img(img), mask(mask), color(color) {}
};

void CalculateEnergy(const cv::Mat& srcMat, cv::Mat& dstMat, cv::Mat& traceMat)
{
    //srcMat是能量矩阵，dstMat是累计能量矩阵（用于DP)，traceMat是轨迹矩阵
    dstMat = srcMat.clone(); //不用“=”，防止两个矩阵指向的都是同一个矩阵，现在只需要传里面的数值   
    for (int i = 1; i < srcMat.rows; i++) { //从第2行开始计算
        //第一列
        if (dstMat.at<float>(i - 1, 0) <= dstMat.at<float>(i - 1, 1)) {
            dstMat.at<float>(i, 0) = srcMat.at<float>(i, 0) + dstMat.at<float>(i - 1, 0);
            traceMat.at<float>(i, 0) = 1; //traceMat记录当前位置的上一行应取那个位置，上左为0，上中1，上右为2
        }
        else {
            dstMat.at<float>(i, 0) = srcMat.at<float>(i, 0) + dstMat.at<float>(i - 1, 1);
            traceMat.at<float>(i, 0) = 2;
        }
        //中间列
        for (int j = 1; j < srcMat.cols - 1; j++) {
            float k[3];
            k[0] = dstMat.at<float>(i - 1, j - 1);
            k[1] = dstMat.at<float>(i - 1, j);
            k[2] = dstMat.at<float>(i - 1, j + 1);
            int index = 0;
            if (k[1] < k[0])
                index = 1;
            if (k[2] < k[index])
                index = 2; 
            dstMat.at<float>(i, j) = srcMat.at<float>(i, j) + dstMat.at<float>(i - 1, j - 1 + index);
            traceMat.at<float>(i,j) = index;
        }
        //最后一列
        if (dstMat.at<float>(i - 1, srcMat.cols - 1) <= dstMat.at<float>(i - 1, srcMat.cols - 2)) {
            dstMat.at<float>(i, srcMat.cols - 1) = srcMat.at<float>(i,srcMat.cols - 1) + dstMat.at<float>(i - 1, srcMat.cols - 1);
            traceMat.at<float>(i, srcMat.cols - 1) = 1; 
        }
        else {
            dstMat.at<float>(i, srcMat.cols - 1) = srcMat.at<float>(i,srcMat.cols - 1) + dstMat.at<float>(i - 1, srcMat.cols - 2);
            traceMat.at<float>(i, srcMat.cols - 1) = 0;
        }

    }
}

// 找出最小能量线
void GetMinEnergyTrace(const cv::Mat& energyMat, const cv::Mat& traceMat, cv::Mat& minTrace)
{
    //enerygyMat是累计能量矩阵，traceMat是轨迹矩阵，minTrace是最小能量路径
    int row = energyMat.rows - 1;// 取的是energyMat最后一行的数据，所以行标是rows-1
    int index = 0;  // 保存的是最小那条轨迹的最下面点在图像中的列标
    // 获得index，即最后那行最小值的位置
    for (int i = 1; i < energyMat.cols; i++) {
        if (energyMat.at<float>(row, i) < energyMat.at<float>(row, index)) {
            index = i;
        }
    }
    // 以下根据traceMat，得到minTrace，minTrace是多行一列矩阵
    minTrace.at<float>(row, 0) = index;
    int tmpIndex = index;
    for (int i = row; i > 0; i--) {
        int temp = traceMat.at<float>(i, tmpIndex);// 当前位置traceMat所存的值
        if (temp == 0) { // 往左走
            tmpIndex = tmpIndex - 1;
        }
        else if (temp == 2) { // 往右走
            tmpIndex = tmpIndex + 1;
        } // 如果temp = 1，则往正上走，tmpIndex不需要做修改
        minTrace.at<float>(i - 1, 0) = tmpIndex;
    }
}

// 删掉一列
void DeleteOneCol(const cv::Mat& srcMat, cv::Mat& dstMat, const cv::Mat& minTrace, cv::Mat& deletedLine)
{
    for (int i = 0; i < dstMat.rows; i++) {
        int k = minTrace.at<float>(i, 0);
        for (int j = 0; j < k; j++)
            dstMat.at<cv::Vec3b>(i, j) = srcMat.at<cv::Vec3b>(i, j);
        for (int j = k; j < dstMat.cols; j++) 
            dstMat.at<cv::Vec3b>(i, j) = srcMat.at<cv::Vec3b>(i, j + 1);
        deletedLine.at<cv::Vec3b>(i, 0) = srcMat.at<cv::Vec3b>(i, k);
    }
}

// 恢复一列
void RecoverOneCol(const cv::Mat& srcMat, cv::Mat& dstMat, const cv::Mat& minTrace, const cv::Mat& deletedLine)
{
    
    cv::Mat recorvedImage(srcMat.rows, srcMat.cols + 1, CV_8UC3);
    for (int i = 0; i < srcMat.rows; i++) {
        int k = minTrace.at<float>(i);
        for (int j = 0; j < k; j++)
            recorvedImage.at<cv::Vec3b>(i, j) = srcMat.at<cv::Vec3b>(i, j);
        recorvedImage.at<cv::Vec3b>(i, k) = deletedLine.at<cv::Vec3b>(i, 0);
        for (int j = k + 1;j < srcMat.cols + 1; j++)
            recorvedImage.at<cv::Vec3b>(i, j) = srcMat.at<cv::Vec3b>(i, j - 1);
    }
    dstMat = recorvedImage.clone();

    //显示恢复的轨迹
    cv::Mat tmpImage = recorvedImage.clone();
    for (int i = 0;i < tmpImage.rows; i++) {
        int k = minTrace.at<float>(i, 0);
        tmpImage.at<cv::Vec3b>(i, k)[0] = 0;
        tmpImage.at<cv::Vec3b>(i, k)[1] = 255;
        tmpImage.at<cv::Vec3b>(i, k)[2] = 0;
    }
    cv::imshow("Seam Carving...", tmpImage);
}

void ShowSeam(const cv::Mat& srcMat, cv::Mat& dstMat, const cv::Mat& minTrace, const cv::Mat& deletedLine) {
    cv::Mat recorvedImage(srcMat.rows, srcMat.cols + 1, CV_8UC3);
    for (int i = 0; i < srcMat.rows; i++) {
        int k = minTrace.at<float>(i);
        for (int j = 0; j < k; j++)
            recorvedImage.at<cv::Vec3b>(i, j) = srcMat.at<cv::Vec3b>(i, j);
        recorvedImage.at<cv::Vec3b>(i, k)[0] = 0;
        recorvedImage.at<cv::Vec3b>(i, k)[1] = 255;
        recorvedImage.at<cv::Vec3b>(i, k)[2] = 0;
        for (int j = k + 1;j < srcMat.cols + 1; j++)
            recorvedImage.at<cv::Vec3b>(i, j) = srcMat.at<cv::Vec3b>(i, j - 1);
    }
    dstMat = recorvedImage.clone();
}

void Shrink(const cv::Mat& image, cv::Mat& outImage, cv::Mat& outMinTrace, cv::Mat& outDeletedLine, const cv::Mat& removeMask, const cv::Mat& remainMask, int kernel)
{
    cv::Mat image_gray(image.rows, image.cols, CV_8U, cv::Scalar(0));
    cv::cvtColor(image, image_gray, CV_BGR2GRAY); //彩色图像转换为灰度图像

    cv::Mat gradiant_H(image.rows, image.cols, CV_32F, cv::Scalar(0)); //水平梯度矩阵
    cv::Mat gradiant_V(image.rows, image.cols, CV_32F, cv::Scalar(0)); //垂直梯度矩阵

    cv::Mat kernel_mine_H = (cv::Mat_<float>(3, 3) << 0, 0, 0, 0, 1, -1, 0, 0, 0); //求水平梯度所使用的卷积核（赋初始值）
    cv::Mat kernel_mine_V = (cv::Mat_<float>(3, 3) << 0, 0, 0, 0, 1, 0, 0, -1, 0); //求垂直梯度所使用的卷积核（赋初始值）

    cv::Mat kernel_Sobel_H = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat kernel_Sobel_V = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    cv::Mat kernel_Laplace_H = (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    cv::Mat kernel_Laplace_V = (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    
    cv::Mat kernel_Roberts_H = (cv::Mat_<float>(2, 2) << 1, 0, 0, -1);
    cv::Mat kernel_Roberts_V = (cv::Mat_<float>(2, 2) << 0, 1, -1, 0);
    
    cv::Mat kernel_H;
    cv::Mat kernel_V;

    //选择kernel
    if (kernel == 0) {
        kernel_H = kernel_mine_H;
        kernel_V = kernel_mine_V;
    }
    else if (kernel == 1) {
        kernel_H = kernel_Sobel_H;
        kernel_V = kernel_Sobel_V;
    }
    else if (kernel == 2) {
        kernel_H = kernel_Laplace_H;
        kernel_V = kernel_Laplace_V;
    }
    else if (kernel == 3) {
        kernel_H = kernel_Roberts_H;
        kernel_V = kernel_Roberts_V;
    }

    cv::filter2D(image_gray, gradiant_H, gradiant_H.depth(), kernel_H);
    cv::filter2D(image_gray, gradiant_V, gradiant_V.depth(), kernel_V);

    cv::Mat gradMag_mat(image.rows, image.rows, CV_32F, cv::Scalar(0));
    cv::add(cv::abs(gradiant_H), cv::abs(gradiant_V), gradMag_mat);//水平与垂直滤波结果的绝对值相加，可以得到近似梯度大小

    for (int i = 0; i < image.rows; i++)  {
        for (int j = 0; j < image.cols; j++)  {
            if (removeMask.at<char>(i, j) == 1) 
                gradMag_mat.at<float>(i, j) -= INF;
            if (remainMask.at<char>(i, j) == 1) 
                gradMag_mat.at<float>(i, j) += INF;
        }
    }

    // 如果要显示梯度大小这个图，因为gradMag_mat深度是CV_32F，所以需要先转换为CV_8U
    cv::Mat testMat;
    gradMag_mat.convertTo(testMat, CV_8U, 1, 0);
    cv::imshow("Gradient", testMat);

    //计算能量线
    cv::Mat energyMat(image.rows, image.cols, CV_32F, cv::Scalar(0));//累计能量矩阵
    cv::Mat traceMat(image.rows, image.cols, CV_32F, cv::Scalar(0));//能量最小轨迹矩阵
    CalculateEnergy(gradMag_mat, energyMat, traceMat); 

    //找出最小能量线
    cv::Mat minTrace(image.rows, 1, CV_32F, cv::Scalar(0));//能量最小轨迹矩阵中的最小的一条的轨迹
    GetMinEnergyTrace(energyMat, traceMat, minTrace);

    //显示最小能量线
    cv::Mat tmpImage = image.clone();
    for (int i = 0; i < image.rows; i++)
    {
        int k = minTrace.at<float>(i, 0);
        tmpImage.at<cv::Vec3b>(i, k)[0] = 0;
        tmpImage.at<cv::Vec3b>(i, k)[1] = 0;
        tmpImage.at<cv::Vec3b>(i, k)[2] = 255;
    }
    cv::imshow("Seam Carving...", tmpImage);

    //删除一列
    cv::Mat image2(image.rows, image.cols - 1, image.type());
    cv::Mat deletedLine(image.rows, 1, CV_8UC3);//记录被删掉的那一列的值
    DeleteOneCol(image, image2, minTrace, deletedLine);
    // cv::imshow("Image Show Window", image2);
    outImage = image2.clone();
    outMinTrace = minTrace.clone();
    outDeletedLine = deletedLine.clone();
}

void Resize(const cv::Mat& image, int delta, int kernel) {
    cv::Mat tmpMat = image.clone();
    cv::Mat traces[MAX_COLS];
    cv::Mat deletedLines[MAX_COLS];
    cv::Mat outImage;
    cv::Mat removeMask(image.rows, image.cols, CV_8U, cv::Scalar(0));
    cv::Mat remainMask(image.rows, image.cols, CV_8U, cv::Scalar(0));
    
    bool shrink = (delta < 0);
    if (shrink) delta = -delta;

    for (int i = 0; i < delta; i++) {
        Shrink(tmpMat, outImage, traces[i], deletedLines[i], removeMask, remainMask, kernel);
        tmpMat = outImage;
        cv::waitKey(50);
    }

    cv::Mat tmpMat2 = outImage.clone();
    cv::Mat tmpMat3 = outImage.clone();//显示Seam
    for (int i = delta - 1; i >= 0; i--) {
        ShowSeam(tmpMat3, outImage, traces[i], deletedLines[i]);
        tmpMat3 = outImage;
    }
    cv::imshow("Seam", tmpMat3);
    cv::imwrite("seam.png", tmpMat3);

    // 放大图片
    if (!shrink) {
        for (int i = delta - 1; i >= 0; i--) {
            for (int j = i + 1; j < delta; j++)  {
                for (int k = 0; k < image.rows; k++) {
                    if (traces[i].at<float>(k, 0) >= traces[j].at<float>(k, 0)) 
                        traces[i].at<float>(k, 0) += 1;
                    else 
                        traces[j].at<float>(k, 0) += 1;
                }
            }
            cv::waitKey(50);
            RecoverOneCol(tmpMat2, outImage, traces[i], deletedLines[i]);
            tmpMat2 = outImage;
            RecoverOneCol(tmpMat2, outImage, traces[i], deletedLines[i]);
            tmpMat2 = outImage;
        }
    }
    cv::imshow("Result", tmpMat2);
    cv::imwrite("result.png", tmpMat2);
}

void OnMouse(int event, int x, int y, int flags, void *param)
{
    MouseArgs *args = (MouseArgs *)param;
    // 按下鼠标左键拖动时
    if ((event == CV_EVENT_MOUSEMOVE || event == CV_EVENT_LBUTTONDOWN) && (flags & CV_EVENT_FLAG_LBUTTON)) {
        int brushRadius = 10;   // 笔刷半径         
        int rows = args->img.rows, cols = args->img.cols;

        // 以下的双重for循环遍历的是半径为10的圆形区域，实现笔刷效果
        for (int i = std::max(0, y - brushRadius); i < std::min(rows, y + brushRadius); i++) {
            int halfChord = sqrt(pow(brushRadius, 2) - pow(i - y, 2)); // 半弦长
            for (int j = std::max(0, x - halfChord); j < std::min(cols, x + halfChord); j++) {
                if (args->mask.at<char>(i, j) == 0) {
                    // 高亮这一笔
                    args->img.at<cv::Vec3b>(i, j) = args->img.at<cv::Vec3b>(i, j) * 0.7 + args->color * 0.3;
                    // 将这一笔添加到选区
                    args->mask.at<char>(i, j) = 1;
                }
            }
        }
    }
}

bool IsRemove(const cv::Mat& image, const cv::Mat& removeMask) {
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++) 
            if (removeMask.at<char>(i, j) == 1) 
                return false;
    return true;
}

void ObjectRemove(const cv::Mat& image, int kernel) {
    cv::Mat showImg = image.clone(); // 拷贝一张图用于显示（因为需要在显示的图上面高亮标注，从而造成修改）
    cv::namedWindow("Remove & Remain", CV_WINDOW_AUTOSIZE); // 新建一个窗口
    cv::Mat removeMask(image.rows, image.cols, CV_8U, cv::Scalar(0)); // 希望获取的待删除选区
    cv::Mat remainMask(image.rows, image.cols, CV_8U, cv::Scalar(0)); // 希望获取的待保留选区

    cv::Vec3b red(0, 0, 255); 
    cv::Vec3b green(0, 255, 0); 
    MouseArgs *args1 = new MouseArgs(showImg, removeMask, green);
    MouseArgs *args2 = new MouseArgs(showImg, remainMask, red);
    cv::setMouseCallback("Remove & Remain", OnMouse, (void*)args1); // 给窗口设置回调函数
    while (1) {
        cv::imshow("Remove & Remain", args1->img);
        // 按 esc 键退出绘图模式，获得选区
        if (cv::waitKey(100) == 27)
            break;
    }
    cv::setMouseCallback("Remove & Remain", OnMouse, (void*)args2); // 给窗口设置回调函数
    while (1) {
        cv::imshow("Remove & Remain", args2->img);
        // 按 esc 键退出绘图模式，获得选区
        if (cv::waitKey(100) == 27)
            break;
    }
    cv::setMouseCallback("Remove & Remain", NULL, NULL); // 取消回调函数
    delete args1; args1 = NULL; // 垃圾回收
    delete args2; args2 = NULL; // 垃圾回收

    cv::Mat tmpMat = image.clone();
    cv::Mat traces[MAX_COLS];
    cv::Mat deletedLines[MAX_COLS];
    cv::Mat outImage;

    int i = 0;
    while (!IsRemove(tmpMat, removeMask)) {
        Shrink(tmpMat, outImage, traces[i], deletedLines[i], removeMask, remainMask, kernel);
        for (int j = 0; j < tmpMat.rows; j++) {
            for (int k = traces[i].at<float>(j, 0); k < tmpMat.cols - 1; k++) {
                removeMask.at<char>(j, k) = removeMask.at<char>(j, k + 1);
                remainMask.at<char>(j, k) = remainMask.at<char>(j, k + 1);
            }
        }
        tmpMat = outImage;
        i++;
        cv::waitKey(50);
    }
    cv::imshow("Result Remove", tmpMat);
    cv::imwrite("result_remove.png", tmpMat);
    cv::Mat tmpMat3 = outImage.clone();//显示Seam
    for (i--; i >= 0; i--) {
        ShowSeam(tmpMat3, outImage, traces[i], deletedLines[i]);
        tmpMat3 = outImage;
    }
    cv::imshow("Seam Remove", tmpMat3);
    cv::imwrite("seam_remove.png", tmpMat3);
}

int main(int argc, char** argv)
{
    int delta, kernel;
    char filename[100];
    std::cout << "Please input filename." << std::endl;
    std::cin >> filename;
    std::cout << "Please input delta." << std::endl;
    std::cin >> delta;
    std::cout << "Please input kernel." << std::endl;
    std::cin >> kernel;

    cv::Mat image = cv::imread(filename);
    cv::namedWindow("Original Image");
    cv::imshow("Original Image", image);
    cv::waitKey(200);
    
    Resize(image, delta, kernel);
    ObjectRemove(image, kernel);

    cv::waitKey(50000);
    return 0;
}
