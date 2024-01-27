#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include<windows.h>
#include <conio.h>
#include <stdlib.h>
#include <time.h>
#include <cctype>
#include <fstream>
#include<direct.h>
#include <filesystem>
#include <cmath>

#include "constantes.h"
#include "structs.h"


using std::cout;
using std::cin;
using std::endl;
using std::array;
using std::vector;
using std::getline;
using std::string;
using namespace cv;
using namespace std;



void criarDiretorio(String diretorio) {

    if (!filesystem::exists(diretorio)) {

        if (filesystem::create_directories(diretorio)) {
            printf("O diretório %s foi criado com êxito.", diretorio.c_str());
        }
        else {
            printf("O diretório %s já existe.", diretorio.c_str());
        }
    }
}

int adicionaFuncionarioFile(String arquivoGravacao) {

    String arquivo = "";
    String nome = "";
    int idFormulario = 0;
    cin >> nome;
    vector<int> ids;

    if (filesystem::exists(arquivoGravacao)) {


        ifstream arquivo(arquivoGravacao);

        if (arquivo.is_open()) {

            string linha;
            while (getline(arquivo, linha)) {

                istringstream ss(linha);


                string parte;
                while (getline(ss, parte, '|')) {


                    if (parte.find("Id") != string::npos) {

                        istringstream idParte(parte);
                        while (getline(idParte, parte, ':')) {

                            if (parte != "Id") {

                                ids.push_back(stoi(parte));
                            }
                        }

                    }
                }

                idFormulario = ids[ids.size() - 1] + 1;

            }
            arquivo.close();
        }


        ofstream arquivoCadastro(arquivoGravacao, ios::app);

        if (arquivoCadastro.is_open()) {

            arquivoCadastro << "Nome:" << nome << "|" << "Id:" << idFormulario << "\n";
            arquivoCadastro.close();
        }

    }
    else {
        ofstream arquivoCadastro(arquivoGravacao);
        arquivoCadastro << "Nome:" << nome << "|" << "Id:" << idFormulario << "\n";
        arquivoCadastro.close();
    }

    return idFormulario;
}

vector<Deteccao> deteccaoSSD(dnn::Net network, Mat frame, int tamanho, float confiancaMinima) {

    int h = frame.rows;
    int w = frame.cols;

    vector<Deteccao> deteccoesPessoa;

    Mat resized;
    resize(frame, resized, Size(tamanho, tamanho), 0, 0, INTER_LINEAR);
    Mat blob = dnn::blobFromImage(resized, 1.0, Size(tamanho, tamanho), Scalar(104.0, 117.0, 123.0));

    network.setInput(blob);

    Mat deteccoes = network.forward();

    Mat_<float> deteccoesMat(deteccoes);

    int num_deteccoes = deteccoes.size[2];

    int idDetec = 0;
    int confiancaRetorno = 0;


    for (int i = 0; i < num_deteccoes; ++i) {
        Vec<float, 7> deteccao = deteccoes.at<Vec<float, 7>>(0, 0, i);

        float confianca = deteccao[2];

        // Verifique se a confiança atende ao limiar
        if (confianca > confiancaMinima) {

            // Extrai as coordenadas da caixa delimitadora
            int startX = static_cast<int>(deteccao[3] * w);
            int startY = static_cast<int>(deteccao[4] * h);
            int endX = static_cast<int>(deteccao[5] * w);
            int endY = static_cast<int>(deteccao[6] * h);



            struct Box box = {
                static_cast<int>(deteccao[3] * w),
                static_cast<int>(deteccao[4] * h),
                static_cast<int>(deteccao[5] * w),
                static_cast<int>(deteccao[6] * h)
            };


            Mat roi = frame(Range(startY, endY), Range(startX, endX));

            Mat roiResized;
            resize(roi, roiResized, Size(60, 80), 0, 0, INTER_LINEAR);

            Mat roiGrey;
            cvtColor(roiResized, roiGrey, COLOR_BGR2GRAY);

            vector<Previsao> previsoes;
            vector<cv::Mat> deteccoes;
            struct Deteccao detec = { idDetec,roiGrey,frame,confianca,box,previsoes,deteccoes,-1.0,0 };


            idDetec += 1;

            deteccoesPessoa.push_back(detec);

            // Desenha a caixa delimitadora na imagem
            rectangle(frame, Point(startX, startY), Point(endX, endY), Scalar(0, 255, 0), 2);
        }
    }

    vector<Deteccao> copiaDeteccoesPessoa = deteccoesPessoa;

    return copiaDeteccoesPessoa;
}


bool validaDeteccao(Deteccao deteccao, float confiancaMinima, int larguraIdeal, int alturaIdeal) {

    bool retorno = false;
    int largura = deteccao.box.endX - deteccao.box.startX;
    int altura = deteccao.box.endY - deteccao.box.startY;

    if (largura > larguraIdeal && altura > alturaIdeal) {

        if (deteccao.confiancaRetorno > confiancaMinima) {

            retorno = true;
        }

    }
    return retorno;

}



vector<Pessoa> comparaPessoasComDeteccao(vector<Pessoa> listaDePessoas, Deteccao deteccao) {

    for (int j = 0; j < listaDePessoas.size(); j++) {

        float x1 = listaDePessoas[j].box.startX;
        float x2 = deteccao.box.startX;

        float y1 = listaDePessoas[j].box.startY;
        float y2 = deteccao.box.startY;

        listaDePessoas[j].distancia = sqrt((pow(x1 - x2, 2) + (pow(y1 - y2, 2))));
    }

    return listaDePessoas;
}


struct Pessoa getPessoaMaisProxima(vector<Pessoa>listaDePessoas) {

    struct Pessoa pessoaMaisProxima = {};

    for (int j = 0; j < listaDePessoas.size(); j++) {

        if (j == 0) {

            pessoaMaisProxima = listaDePessoas[j];
        }
        else {
            if (listaDePessoas[j].distancia < pessoaMaisProxima.distancia) {

                pessoaMaisProxima = listaDePessoas[j];
            }
        }
    }

    return pessoaMaisProxima;
}


struct Deteccao getDeteccaoMaisProxima(vector<Deteccao>listaDedeteccoes) {

    struct Deteccao deteccaoMaisProxima = {};

    for (int j = 0; j < listaDedeteccoes.size(); j++) {

        if (j == 0) {

            deteccaoMaisProxima = listaDedeteccoes[j];
        }
        else {
            if (listaDedeteccoes[j].distancia < deteccaoMaisProxima.distancia) {

                deteccaoMaisProxima = listaDedeteccoes[j];
            }
        }
    }

    return deteccaoMaisProxima;
}


void mostrarPessoaDetectada(Pessoa pessoaMaisProxima, Deteccao deteccao) {

    if (pessoaMaisProxima.distancia < 20 && pessoaMaisProxima.distancia != -1) {

        HersheyFonts font = FONT_HERSHEY_COMPLEX;

        double fontScale = 0.5;

        Point org = Point(pessoaMaisProxima.box.startX, pessoaMaisProxima.box.startY - 10);

        pessoaMaisProxima.box = deteccao.box;


        putText(deteccao.imagem, pessoaMaisProxima.nome, org, font, fontScale, CV_RGB(118, 185, 0), 1, 8, false);
    }

}