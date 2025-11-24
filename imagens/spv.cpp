/******************************************************
 * Nome do Programa: SPV - Sistema de Processamento Visual
 * Equipe: ROADWATCH
 * Integrantes:
 *  - Fernanda Ayumi - RA: 11202321172
 *  - Ingrid Xavier  - RA: 11202130019
 *  - Gabriel Rothen - RA: 11202321586
 *
 * Data: 2025
 * Descrição:
 *   Detecção de movimento aprimorada e estável.
 *   Ignora ruído, sombras e pequenas variações de luz.
 *   Salva imagem completa da webcam após 1s contínuo de movimento real.
 *   Exibe alerta visual e sonoro.
 *
 * Execução:
 *   $ mkdir build && cd build
 *   $ cmake ..
 *   $ make
 *   $ ./spv
 ******************************************************/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <ctime>

using namespace std;
using namespace cv;
using namespace chrono;

int main() {
    string nomeEquipe = "ROADWATCH";
    string tituloJanela = "SPV - Sistema de Processamento Visual - " + nomeEquipe;

    // ==== MENU INICIAL ====
    Mat menu = Mat::zeros(Size(720, 420), CV_8UC3);
    menu.setTo(Scalar(0,0,0)); // fundo preto
    rectangle(menu, Point(0,0), Point(menu.cols, 80), Scalar(255,0,0), FILLED); // barra azul
    putText(menu, "SISTEMA DE PROCESSAMENTO VISUAL", Point(30, 55),
            FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
    putText(menu, "Equipe: " + nomeEquipe, Point(30, 130),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,0,0), 2);
    putText(menu, "Pressione 'i' para iniciar a webcam", Point(30, 210),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,0,0), 2);
    putText(menu, "Pressione 'q' para sair", Point(30, 260),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,0,0), 2);

    namedWindow("MENU - " + nomeEquipe, WINDOW_AUTOSIZE);
    imshow("MENU - " + nomeEquipe, menu);

    char tecla;
    while (true) {
        tecla = (char)waitKey(0);
        if (tecla == 'i') {
            destroyWindow("MENU - " + nomeEquipe);
            break;
        }
        if (tecla == 'q' || tecla == 27) {
            destroyAllWindows();
            return 0;
        }
    }

    // ==== ABRE WEBCAM ====
    VideoCapture cam(0);
    if (!cam.isOpened()) {
        cerr << "Erro ao acessar webcam!" << endl;
        return -1;
    }

    cam.set(CAP_PROP_FRAME_WIDTH, 640);
    cam.set(CAP_PROP_FRAME_HEIGHT, 480);

    Ptr<BackgroundSubtractorMOG2> bgSub = createBackgroundSubtractorMOG2();
    bgSub->setVarThreshold(25);
    bgSub->setHistory(400);
    bgSub->setDetectShadows(true);

    cout << "Webcam iniciada. Detectando movimento real..." << endl;
    cout << "Imagem será salva automaticamente após 1s contínuo.\n" << endl;

    Mat frame, mask;
    bool detectado = false, salvo = false;
    steady_clock::time_point inicio;
    bool ativo = false;

    int framesComMovimento = 0;
    int framesSemMovimento = 0;
    const int LIMITE_FRAMES = 5;  // estabilidade temporal

    while (true) {
        cam >> frame;
        if (frame.empty()) break;

        // ROI central para reduzir falsos positivos nas bordas
        Rect roi(frame.cols/6, frame.rows/6, frame.cols*2/3, frame.rows*2/3);
        Mat frameROI = frame(roi);

        Mat blur;
        GaussianBlur(frameROI, blur, Size(5,5), 0);

        bgSub->apply(blur, mask, 0.005);

        // pós-processamento pesado contra ruído
        threshold(mask, mask, 200, 255, THRESH_BINARY);
        morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
        morphologyEx(mask, mask, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(7,7)));
        dilate(mask, mask, getStructuringElement(MORPH_RECT, Size(5,5)), Point(-1,-1), 2);

        vector<vector<Point>> contornos;
        findContours(mask, contornos, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        detectado = false;
        Rect maiorObj;
        double maiorArea = 0;

        for (auto &c : contornos) {
            double area = contourArea(c);
            if (area < 2500 || area > 50000) continue; // ignora muito pequeno/grande
            Rect r = boundingRect(c);
            double ratio = (double)r.width / (double)r.height;
            if (ratio < 0.3 || ratio > 3.5) continue;
            if (area > maiorArea) {
                maiorArea = area;
                maiorObj = r;
                detectado = true;
            }
        }

        // sistema de estabilidade temporal
        if (detectado) {
            framesComMovimento++;
            framesSemMovimento = 0;
        } else {
            framesSemMovimento++;
            if (framesSemMovimento > 5) framesComMovimento = 0;
        }

        bool movimentoReal = (framesComMovimento > LIMITE_FRAMES);

        if (movimentoReal) {
            Rect ajustado(maiorObj.x + roi.x, maiorObj.y + roi.y, maiorObj.width, maiorObj.height);
            rectangle(frame, ajustado, Scalar(255,0,0), 2);
            putText(frame, "MOVIMENTO DETECTADO!", Point(20, 40),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,0,0), 2);

            if (!ativo) {
                inicio = steady_clock::now();
                ativo = true;
                salvo = false;
            }

            double duracao = duration_cast<seconds>(steady_clock::now() - inicio).count();
            double progresso = min(duracao / 1.0, 1.0);
            int largura = (int)(frame.cols * progresso);
            rectangle(frame, Point(0, frame.rows - 10), Point(largura, frame.rows),
                      Scalar(255,0,0), FILLED);

            // === SALVAR FRAME COMPLETO ===
            if (duracao >= 1.0 && !salvo) {
                string nomeArquivo = "movimento_" + to_string(time(0)) + ".jpg";
                imwrite(nomeArquivo, frame);
                cout << "\a"; // beep sonoro no terminal
                cout << "[SALVO] Movimento detectado - Imagem completa salva: " << nomeArquivo << endl;

                // Exibir pop-up com a imagem completa
                Mat alerta = frame.clone();
                putText(alerta, "ALERTA - MOVIMENTO DETECTADO!", Point(30, 50),
                        FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,255), 3);
                namedWindow("ALERTA - " + nomeEquipe, WINDOW_AUTOSIZE);
                imshow("ALERTA - " + nomeEquipe, alerta);

                salvo = true;
            }
        } else {
            ativo = false;
            destroyWindow("ALERTA - " + nomeEquipe);
        }

        // bordas e instruções
        rectangle(frame, Point(2,2), Point(frame.cols-2, frame.rows-2), Scalar(255,0,0), 2);
        putText(frame, "Pressione 'q' para sair", Point(20, frame.rows - 20),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255), 1);

        imshow(tituloJanela, frame);
        tecla = (char)waitKey(20);
        if (tecla == 'q' || tecla == 27) break;
    }

    cam.release();
    destroyAllWindows();
    cout << "\nExecução encerrada. Obrigado por utilizar o SPV - " << nomeEquipe << "!" << endl;
    return 0;
}

