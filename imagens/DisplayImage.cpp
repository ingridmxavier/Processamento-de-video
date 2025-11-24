#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

int main() {
    VideoCapture webcam(0);
    if (!webcam.isOpened()) {
        cerr << "❌ Erro ao abrir a webcam!" << endl;
        return -1;
    }

    cout << "✅ Webcam iniciada. Pressione 'q' para sair." << endl;

    Ptr<BackgroundSubtractorMOG2> bgSubtractor = createBackgroundSubtractorMOG2();
    bgSubtractor->setDetectShadows(true);

    Mat frame, gray, fgMask, frameAnteriorGray;
    vector<Point2f> pontosAntigos, pontosNovos;

    int movimentoConsecutivo = 0;
    bool popupAberto = false;
    static bool avisoEnviado = false;

    while (true) {
        webcam >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        bgSubtractor->apply(frame, fgMask, 0.005);
        threshold(fgMask, fgMask, 200, 255, THRESH_BINARY);
        erode(fgMask, fgMask, Mat(), Point(-1, -1), 1);
        dilate(fgMask, fgMask, Mat(), Point(-1, -1), 2);

        vector<vector<Point>> contornos;
        findContours(fgMask, contornos, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        bool movimentoDetectado = false;
        Rect maiorObjeto;
        double maiorArea = 0;

        // 🎯 Detecta o maior objeto
        for (const auto& cnt : contornos) {
            double area = contourArea(cnt);
            if (area > 3000 && area > maiorArea) {
                maiorArea = area;
                maiorObjeto = boundingRect(cnt);
                movimentoDetectado = true;
            }
        }

        if (movimentoDetectado) {
            rectangle(frame, maiorObjeto, Scalar(0, 255, 0), 2);
            putText(frame, "OBJETO DETECTADO", maiorObjeto.tl(),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

            if (!frameAnteriorGray.empty()) {
                vector<uchar> status;
                vector<float> err;

                goodFeaturesToTrack(frameAnteriorGray, pontosAntigos, 200, 0.01, 10);
                if (!pontosAntigos.empty()) {
                    calcOpticalFlowPyrLK(frameAnteriorGray, gray, pontosAntigos,
                                         pontosNovos, status, err);

                    int movimentoReal = 0;
                    for (size_t i = 0; i < status.size(); i++) {
                        if (status[i] &&
                            norm(pontosNovos[i] - pontosAntigos[i]) > 3.0)
                            movimentoReal++;
                    }

                    if (movimentoReal > 10)
                        movimentoConsecutivo++;
                    else if (movimentoConsecutivo > 0)
                        movimentoConsecutivo--;
                }
            }
        }

        frameAnteriorGray = gray.clone();

        // ⚠️ ALERTA + FOTO
        if (movimentoConsecutivo >= 3) {

            if (!popupAberto) {
                namedWindow("⚠️ ALERTA!", WINDOW_NORMAL);
                popupAberto = true;
            }

            if (!avisoEnviado) {
                cout << "🚨 OBJETO DETECTADO DE VERDADE!" << endl;
                avisoEnviado = true;
            }

            Rect box = maiorObjeto & Rect(0, 0, frame.cols, frame.rows);
            if (box.area() > 0) {
                Mat objeto = frame(box).clone();

                // 📸 Salvando imagem
                string nomeArquivo = "objeto_" + to_string(time(0)) + ".jpg";
                imwrite(nomeArquivo, objeto);
                cout << "📸 Foto salva: " << nomeArquivo << endl;

                resize(objeto, objeto, Size(300, 300));
                imshow("⚠️ ALERTA!", objeto);
            }

        } else {
            if (popupAberto) {
                destroyWindow("⚠️ ALERTA!");
                popupAberto = false;
            }
            avisoEnviado = false;
        }

        imshow("Deteccao Avancada", frame);

        char tecla = (char)waitKey(20);
        if (tecla == 'q' || tecla == 27) break;
    }

    webcam.release();
    destroyAllWindows();
    return 0;
}

