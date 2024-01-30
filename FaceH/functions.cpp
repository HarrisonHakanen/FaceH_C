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
#include <windows.h>
#include <conio.h>
#include <stdlib.h>
#include <time.h>
#include <cctype>
#include <fstream>
#include <direct.h>
#include <filesystem>
#include <cmath>


#include "functions.h"
#include "global.h"




void criarDiretorio(std::string diretorio) {

    if (!std::filesystem::exists(diretorio)) {

        if (std::filesystem::create_directories(diretorio)) {
            printf("O diretório %s foi criado com êxito.", diretorio.c_str());
        }
        else {
            printf("O diretório %s já existe.", diretorio.c_str());
        }
    }
}

int adicionaFuncionarioFile(std::string arquivoGravacao) {

    std::string arquivo = "";
    std::string nome = "";
    int idFormulario = 0;
    std::cin >> nome;
    std::vector<int> ids;

    if (std::filesystem::exists(arquivoGravacao)) {


        std::ifstream arquivo(arquivoGravacao);

        if (arquivo.is_open()) {

            std::string linha;
            while (getline(arquivo, linha)) {

                std::istringstream ss(linha);


                std::string parte;
                while (getline(ss, parte, '|')) {


                    if (parte.find("Id") != std::string::npos) {

                        std::stringstream idParte(parte);
                        while (std::getline(idParte, parte, ':')) {

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


        std::ofstream arquivoCadastro(arquivoGravacao, std::ios::app);

        if (arquivoCadastro.is_open()) {

            arquivoCadastro << "Id:" << idFormulario << "|" << "Nome:" << nome << "\n";
            arquivoCadastro.close();
        }

    }
    else {
        std::ofstream arquivoCadastro(arquivoGravacao);
        arquivoCadastro << "Id:" << idFormulario << "|" << "Nome:" << nome << "\n";
        arquivoCadastro.close();
    }

    return idFormulario;
}

std::vector<Deteccao> deteccaoSSD(cv::dnn::Net network, cv::Mat frame, int tamanho, float confiancaMinima) {

    int h = frame.rows;
    int w = frame.cols;

    std::vector<Deteccao> deteccoesPessoa;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(tamanho, tamanho), 0, 0, cv::INTER_LINEAR);
    cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0, cv::Size(tamanho, tamanho), cv::Scalar(104.0, 117.0, 123.0));

    network.setInput(blob);

    cv::Mat deteccoes = network.forward();

    cv::Mat_<float> deteccoesMat(deteccoes);

    int num_deteccoes = deteccoes.size[2];

    int idDetec = 0;
    int confiancaRetorno = 0;


    for (int i = 0; i < num_deteccoes; ++i) {
        cv::Vec<float, 7> deteccao = deteccoes.at<cv::Vec<float, 7>>(0, 0, i);

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


            cv::Mat roi = frame(cv::Range(startY, endY), cv::Range(startX, endX));

            cv::Mat roiResized;
            cv::resize(roi, roiResized, cv::Size(60, 80), 0, 0, cv::INTER_LINEAR);

            cv::Mat roiGrey;
            cv::cvtColor(roiResized, roiGrey, cv::COLOR_BGR2GRAY);

            std::vector<Previsao> previsoes;

            //O rosto em escala de cinza é a primeira deteção do objeto Deteccao
            //Ou seja, sempre que quando uma detecção existir, ela já vai ter por padrão um rosto nas detecções.
            std::vector<cv::Mat> deteccoes;
            deteccoes.push_back(roiGrey);

            Pessoa pessoa = { -1,"",box,0 };

            struct Deteccao detec = { idDetec,frame,confianca,box,previsoes,deteccoes,pessoa,-1.0,0,false,0 };


            idDetec += 1;

            deteccoesPessoa.push_back(detec);

            // Desenha a caixa delimitadora na imagem
            cv::rectangle(frame, cv::Point(startX, startY), cv::Point(endX, endY), cv::Scalar(0, 255, 0), 2);
        }
    }

    std::vector<Deteccao> copiaDeteccoesPessoa = deteccoesPessoa;

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


void mostrarPessoaDetectada(Pessoa pessoaMaisProxima, Deteccao deteccao) {

    if (pessoaMaisProxima.distancia < distanciaMinima && pessoaMaisProxima.distancia != -1) {

        cv::HersheyFonts font = cv::FONT_HERSHEY_COMPLEX;

        double fontScale = 0.5;

        cv::Point org = cv::Point(pessoaMaisProxima.box.startX, pessoaMaisProxima.box.startY - 10);

        pessoaMaisProxima.box = deteccao.box;


        putText(deteccao.imagem, pessoaMaisProxima.nome, org, font, fontScale, CV_RGB(118, 185, 0), 1, 8, false);
    }

}



std::vector<Deteccao> calculaDistanciaDeteccaoParaDeteccoesGlobais(std::vector<Deteccao> deteccoesGlobais, Deteccao deteccao) {

    //dgi = deteccoes globais index
    for (int dgi = 0; dgi < deteccoesGlobais.size(); dgi++) {

        float x1 = deteccoesGlobais[dgi].box.startX;
        float x2 = deteccao.box.startX;

        float y1 = deteccoesGlobais[dgi].box.startY;
        float y2 = deteccao.box.startY;

        deteccoesGlobais[dgi].distancia = sqrt((pow(x1 - x2, 2) + (pow(y1 - y2, 2))));
    }

    return deteccoesGlobais;
}

Deteccao getDeteccaoMaisProxima(std::vector<Deteccao> deteccoesGlobais) {

    struct Deteccao deteccaoMaisProxima = {};

    for (int dgi = 0; dgi < deteccoesGlobais.size(); dgi++) {

        if (dgi == 0) {
            deteccaoMaisProxima = deteccoesGlobais[dgi];
        }
        else {
            if (deteccaoMaisProxima.distancia < deteccoesGlobais[dgi].distancia) {

                deteccaoMaisProxima = deteccoesGlobais[dgi];
            }
        }
    }
    return deteccaoMaisProxima;
}


std::vector<Deteccao> validaDeteccaoMaisProxima(std::vector<Deteccao>deteccoesGlobais, Deteccao deteccaoMaisProxima, Deteccao deteccaoAtual) {

    if (deteccaoMaisProxima.distancia < distanciaMinima && deteccaoMaisProxima.distancia != -1) {

        for (int dgi = 0; dgi < deteccoesGlobais.size(); dgi++) {


            //Verifica se a detecção está esperando, caso ela esteja vai sendo decrementado 1 no tempo de espera até o tempo zerar,
            //quando isso acontecer a detecção já não estará mais esperando.
            if (deteccoesGlobais[dgi].esperando == false) {

                if (deteccoesGlobais[dgi].idDeteccao == deteccaoMaisProxima.idDeteccao) {

                    //Atribui o rosto a detecção que esta na lista de deteccoesGlobais
                    //E também é atualizado a posição atual do rosto
                    deteccoesGlobais[dgi].deteccoes.push_back(deteccaoMaisProxima.deteccoes[0]);
                    deteccoesGlobais[dgi].box = deteccaoAtual.box;
                }
            }
            else {

                if (deteccoesGlobais[dgi].tempoEspera > 0) {

                    deteccoesGlobais[dgi].tempoEspera -= 1;
                }
                else {

                    deteccoesGlobais[dgi].esperando = false;
                }
            }


        }
    }
    /*
    else {
        qtdVezesQueCriouDeteccao += 1;
        deteccaoMaisProxima.idDeteccao = deteccoesGlobais[deteccoesGlobais.size() - 1].idDeteccao + 1;
        deteccoesGlobais.push_back(deteccaoMaisProxima);
    }
    */

    return deteccoesGlobais;

}


std::vector<int> retornaPrevisoes(Deteccao deteccaoGlobal) {

    std::vector<int> previsoes;

    cv::Ptr <cv::face::FaceRecognizer> lbphClassifier = cv::face::LBPHFaceRecognizer::create();


    if (std::filesystem::exists(modeloYml)) {

        lbphClassifier->read(modeloYml);

        for (int rostoIndex = 0; rostoIndex < deteccaoGlobal.deteccoes.size(); rostoIndex++) {

            int prev = lbphClassifier->predict(deteccaoGlobal.deteccoes[rostoIndex]);

            previsoes.push_back(prev);
        }

    }

    return previsoes;
}


Recorrencia getRecorrenciasDeClassificacao(std::vector<int>previsoes) {

    std::vector<int> classificacoes;
    std::vector<int> qtdVezes;


    for (int prevIndex = 0; prevIndex < previsoes.size(); prevIndex++) {

        int qtd = 0;
        bool achou = false;
        bool verificou = false;
        for (int subIndex = prevIndex; subIndex < previsoes.size(); subIndex++) {

            if (!verificou) {

                for (int classIndex = 0; classIndex < classificacoes.size(); classIndex++) {

                    if (classificacoes[classIndex] == previsoes[subIndex]) {
                        achou = true;
                    }
                }

                if (!achou) {

                    classificacoes.push_back(previsoes[subIndex]);
                    qtd += 1;
                }

                verificou = true;
            }
            else {

                if (previsoes[prevIndex] == previsoes[subIndex]) {
                    qtd += 1;
                }

            }
        }

        if (!achou) {
            qtdVezes.push_back(qtd);
        }
    }

    Recorrencia recorrencia = { classificacoes ,qtdVezes };
    return recorrencia;

}


int getIndexDaClassificacaoMaisRecorrente(Recorrencia recorrencia) {

    int indexMaior = 0;
    int maior = 0;
    for (int qtdIndex = 0; qtdIndex < recorrencia.qtdVezes.size(); qtdIndex++) {
        if (qtdIndex == 0) {
            maior = recorrencia.qtdVezes[qtdIndex];
            indexMaior = qtdIndex;
        }
        else {
            if (recorrencia.qtdVezes[qtdIndex] > maior) {
                maior = recorrencia.qtdVezes[qtdIndex];
                indexMaior = qtdIndex;
            }
        }
    }
    return indexMaior;
}


Pessoa getPessoaPelaClassificacao(Recorrencia recorrencia, int indexMaior) {

    int maiorClassificacao = recorrencia.classificacoes[indexMaior];
    int confiancaClassificacao = recorrencia.qtdVezes[indexMaior] / qtdDeDeteccoesPorCluster;

    Box box = {};
    Pessoa pessoaEncontrada = { -1,"",box,-1 };

    if (confiancaClassificacao > confiancaMinimaDaPrevisao) {

        if (std::filesystem::exists(arquivoGravacao)) {

            std::ifstream arquivo(arquivoGravacao);

            if (arquivo.is_open()) {

                std::string linha;


                int idPessoa = -1;
                std::string nome = "";
                bool encontrouPessoa = false;

                while (getline(arquivo, linha)) {

                    std::istringstream ss(linha);

                    std::string parte;


                    if (!encontrouPessoa) {

                        while (getline(ss, parte, '|')) {

                            //Pega o Id da pessoa
                            if (parte.find("Id") != std::string::npos) {

                                std::istringstream idParte(parte);
                                while (getline(idParte, parte, ':')) {

                                    if (parte != "Id") {

                                        if (maiorClassificacao == stoi(parte)) {
                                            idPessoa = stoi(parte);
                                            encontrouPessoa = true;
                                        }

                                        std::cout << parte;
                                    }
                                }
                            }

                            if (encontrouPessoa) {

                                //Pega o Nome da pessoa caso tenha encontrado ela pelo id
                                if (parte.find("Nome") != std::string::npos) {

                                    std::istringstream idParte(parte);
                                    while (std::getline(idParte, parte, ':')) {

                                        if (parte != "Nome") {

                                            nome = parte;
                                            std::cout << parte;
                                        }
                                    }
                                }
                            }
                        }
                    }

                }

                Box box = {};

                pessoaEncontrada.id = idPessoa;
                pessoaEncontrada.nome = nome;
                pessoaEncontrada.distancia = -1;
                pessoaEncontrada.box = box;
            }
        }

    }

    return pessoaEncontrada;

}


void escreveNome(Deteccao deteccaoGlobal, Deteccao deteccaoSSD, cv::InputOutputArray frame) {

    if (deteccaoGlobal.pessoa.id != -1) {

        cv::putText(
            frame,
            deteccaoGlobal.pessoa.nome,
            cv::Point(deteccaoSSD.box.startX, deteccaoSSD.box.startY),
            cv::FONT_HERSHEY_DUPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            2,
            false);
    }
}