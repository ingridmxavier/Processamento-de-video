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
 *   Integração com Telegram para envio de mensagens e imagens.
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
#include <curl/curl.h>  // Para integração com o Telegram

using namespace std;
using namespace cv;
using namespace chrono;

// Função para enviar uma mensagem via Telegram
void enviarMensagemTelegram(const string& botToken, const string& chatID, const string& mensagem) {
    CURL *curl;
    CURLcode res;

    string url = "https://api.telegram.org/bot" + botToken + "/sendMessage?chat_id=" + chatID + "&text=" + mensagem;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        res = curl_easy_perform(curl);
        
        if (res != CURLE_OK) {
            cerr << "Erro ao enviar mensagem: " << curl_easy_strerror(res) << endl;
        }
        
        curl_easy_cleanup(curl);
    }
    
    curl_global_cleanup();
}

// Função para enviar uma imagem via Telegram
void enviarFotoTelegram(const string& botToken, const string& chatID, const string& caminhoImagem) {
    CURL *curl;
    CURLcode res;
    FILE *file;
    
    string url = "https://api.telegram.org/bot" + botToken + "/sendPhoto?chat_id=" + chatID;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    
    if (curl) {
        file = fopen(caminhoImagem.c_str(), "rb");
        if (!file) {
            cerr << "Erro ao abrir a imagem!" << endl;
            return;
        }
        
        struct curl_httppost *formpost = NULL;
        struct curl_httppost *lastptr = NULL;
        
        curl_formadd(&formpost, &lastptr,
                     CURLFORM_COPYNAME, "photo",
                     CURLFORM_FILE, caminhoImagem.c_str(),
                     CURLFORM_END);
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
        
        res = curl_easy_perform(curl);
        
        if (res != CURLE_OK) {
            cerr << "Erro ao enviar imagem: " << curl_easy_strerror(res) << endl;
        }
        
        curl_easy_cleanup(curl);
        curl_global_cleanup();
        fclose(file);
    }
}

// Função para gerar o link do Telegram
void gerarLinkTelegram(const string& botUsername, const string& mensagem, const string& caminhoImagem) {
    string url = "https://web.telegram.org/k/#@" + botUsername;
    string mensagemCompleta = mensagem + "\nImagem salva: " + caminhoImagem;
    
    // Exibir o link direto para o Telegram
    cout << "Clique no link abaixo para abrir a conversa com o bot e visualizar a mensagem:" << endl;
    cout << url << endl;
    cout << "Mensagem: " << mensagemCompleta << endl;
}

int main() {
    string nomeEquipe = "ROADWATCH";
    string tituloJanela = "SPV - Sistema de Processamento Visual - " + nomeEquipe;

    // Token e Chat ID do Telegram
    string botToken = "8434266352:AAF0iZzvc9eTDMM2wwB-I8os121p3bC3vGI";
    string chatID = "8540694767";  // Substitua com seu chat ID correto

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

                // Enviar mensagem para o Telegram
                enviarMensagemTelegram(botToken, chatID, "Movimento detectado!");
                enviarFotoTelegram(botToken, chatID, nomeArquivo);

                salvo = true;
            }
        } else {
            ativo = false;
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

