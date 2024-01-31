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
#include "funcoes.h"


using std::cout;
using std::cin;
using std::endl;
using std::array;
using std::vector;
using std::getline;
using std::string;
using namespace cv;
using namespace std;





//Funções
void criarDiretorio(String diretorio);
int adicionaFuncionarioFile(String arquivoGravacao);
vector<Deteccao> deteccaoSSD(dnn::Net network, Mat frame, int tamanho, float confiancaMinima);
bool validaDeteccao(Deteccao deteccao, float confiancaMinima, int larguraIdeal, int alturaIdeal);
vector<Pessoa> comparaPessoasComDeteccao(vector<Pessoa> listaDePessoas, Deteccao deteccao);
struct Pessoa getPessoaMaisProxima(vector<Pessoa>listaDePessoas);
void mostrarPessoaDetectada(Pessoa pessoaMaisProxima, Deteccao deteccao);
struct Deteccao getDeteccaoMaisProxima(vector<Deteccao>listaDedeteccoes);


vector<Deteccao> ultimasDeteccoes;
vector<Pessoa> listaDePessoas;
int tempoPreverProximoLimite;
int tempoPreverProximo;
int qtdFramesPrevisaoTotal;
int qtdFramesPrevisao;
int qtdFramesCadastroTotal;
int qtdFramesCadastro;


int main()
{

    criarDiretorio(modeloSSDPath);
    criarDiretorio(pessoasCadastradasPath);
    criarDiretorio(facesPath);
    criarDiretorio(modelosYmlPath);


    int escolha = 0;



    while (escolha != 4) {

        printf("Bem vindo ao FaceH");
        printf("Selecione uma opção:\n1 - Adicionar pessoa\n2 - Listar pessoas cadastradas\n3 - Continuar sistema\n4 - Sair do sistema\n\n");

        cin >> escolha;


        if (escolha == 1) {



            system("cls");
            printf("Adicionar pessoa\n");

            int ultimoId = adicionaFuncionarioFile(arquivoGravacao);

            printf("Preparando câmera para reconhecimento...");


            bool continuaGravacao = true;
            vector<int>idList;
            vector<Mat> faceList;

            Mat frame;

            VideoCapture cap;



            int deviceID = 0;
            int apiID = CAP_ANY;

            cap.open(deviceID, apiID);

            if (!cap.isOpened()) {
                cerr << "ERROR! Camera inacessível\n";
                return -1;
            }

            cout << "Start grabbing" << endl
                << "Press any key to terminate" << endl;





            while (continuaGravacao)
            {

                cap.read(frame);

                vector<Deteccao> deteccoes = deteccaoSSD(network, frame, 300, 0.9f);

                //Inicio função

                for (int i = 0; i < deteccoes.size(); i++) {

                    if (validaDeteccao(deteccoes[i], confiancaMinimaDeteccao, larguraIdeal, alturaIdeal)) {


                        imshow("Câmera", deteccoes[i].imagem);

                        idList.push_back(ultimoId);
                        faceList.push_back(deteccoes[i].rosto);

                        qtdFramesCadastro += 1;


                        if (qtdFramesCadastro > qtdFramesCadastroTotal) {
                            continuaGravacao = false;
                            qtdFramesCadastro = 0;
                        }
                    }
                    else {
                        imshow("Câmera", frame);
                    }


                }
                imshow("Câmera", frame);

                //Fim função

                if (waitKey(5) >= 0)
                    break;

            }

            cap.release();
            destroyAllWindows();


            //Inicio função            

            Ptr <face::FaceRecognizer> lbphClassifier = face::LBPHFaceRecognizer::create();


            if (filesystem::exists(modeloYml)) {

                lbphClassifier->read(modeloYml);
                lbphClassifier->update(faceList, idList);

            }
            else {

                lbphClassifier->train(faceList, idList);

            }

            lbphClassifier->write(modeloYml);


            //Fim função

        }

        else if (escolha == 2) {

            system("cls");
            printf("Listando pessoas\n");
        }

        else if (escolha == 3) {

            system("cls");
            cout << "Continuar sistema\n";


            bool identificouFace = false;
            vector<Box>imagensLista;
            vector<float> previsoes;
            vector<Previsao>listaPrevisoesObj;


            Mat frame;

            VideoCapture cap;

            int deviceID = 0;
            int apiID = CAP_ANY;

            cap.open(deviceID, apiID);


            while (true) {

                cap.read(frame);

                vector<Deteccao> deteccoes = deteccaoSSD(network, frame, 300, 0.9f);


                for (int i = 0; i < deteccoes.size(); i++) {

                    if (validaDeteccao(deteccoes[i], confiancaMinimaDeteccao, larguraIdeal, alturaIdeal)) {


                        rectangle(
                            frame,
                            Point(deteccoes[i].box.startX, deteccoes[i].box.startY),
                            Point(deteccoes[i].box.endX, deteccoes[i].box.endY),
                            Scalar(0, 255, 0), 2);


                        if (listaDePessoas.size() > 0) {


                            //Calcula a distancia das pessoas com a deteção atual
                            listaDePessoas = comparaPessoasComDeteccao(listaDePessoas, deteccoes[i]);


                            //Pega a pessoa mais próxima da detecção                            
                            struct Pessoa pessoaMaisProxima = getPessoaMaisProxima(listaDePessoas);


                            //Mostra o nome da pessoa mais próxima na tela
                            mostrarPessoaDetectada(pessoaMaisProxima, deteccoes[i]);
                        }

                        if (deteccoes[i].confiancaRetorno > 0.9 && tempoPreverProximo == 0) {

                            if (ultimasDeteccoes.size() > 0) {

                                struct Deteccao deteccaoMaisProxima = getDeteccaoMaisProxima(ultimasDeteccoes);

                                if (deteccaoMaisProxima.distancia < 20) {

                                    identificouFace = false;
                                    Ptr <face::FaceRecognizer> lbphClassifier = face::LBPHFaceRecognizer::create();

                                    float previsao = 0;
                                    if (filesystem::exists(modeloYml)) {

                                        lbphClassifier->read(modeloYml);
                                        previsao = lbphClassifier->predict(deteccoes[i].rosto);
                                    }

                                    vector<float> prevList;
                                    prevList.push_back(previsao);
                                    struct Previsao prev = { 0, prevList };

                                    deteccaoMaisProxima.previsoes.push_back(prev);

                                    previsoes.push_back(previsao);
                                    imagensLista.push_back(deteccoes[i].box);

                                    qtdFramesPrevisao += 1;



                                    vector<float>listaDeMedias;
                                    if (listaPrevisoesObj.size() > 0) {
                                        for (int j = 0; j < listaPrevisoesObj.size(); j++) {

                                            int somatoriaDiferencas = 0;

                                            for (int k = 0; k < listaPrevisoesObj[j].previsao.size(); k++) {

                                                somatoriaDiferencas += listaPrevisoesObj[j].previsao[k] - previsao;
                                            }
                                            listaDeMedias.push_back(somatoriaDiferencas / listaPrevisoesObj[j].previsao.size());
                                        }
                                    }
                                    else {

                                        vector<float> prevList;
                                        prevList.push_back(previsao);
                                        struct Previsao prevNew = { 0,prevList };

                                        listaPrevisoesObj.push_back(prevNew);

                                        if (qtdFramesPrevisao == qtdFramesPrevisaoTotal) {
                                            tempoPreverProximo = tempoPreverProximoLimite;
                                            qtdFramesPrevisao = 0;
                                        }
                                    }
                                }
                                else {

                                    identificouFace = false;

                                    Ptr <face::FaceRecognizer> lbphClassifier = face::LBPHFaceRecognizer::create();

                                    float previsao = 0;
                                    if (filesystem::exists(modeloYml)) {

                                        lbphClassifier->read(modeloYml);
                                        previsao = lbphClassifier->predict(deteccoes[i].rosto);
                                    }

                                    vector<float> prevList;
                                    prevList.push_back(previsao);
                                    struct Previsao prev = { 0, prevList };

                                    deteccoes[i].previsoes.push_back(prev);

                                    previsoes.push_back(previsao);
                                    imagensLista.push_back(deteccoes[i].box);


                                    //Pega maior detecção
                                    int maiorId = 0;

                                    for (int j = 0; j < ultimasDeteccoes.size(); j++) {

                                        if (j == 0) {
                                            maiorId = ultimasDeteccoes[j].idDeteccao;
                                        }
                                        if (ultimasDeteccoes[j].idDeteccao > maiorId) {
                                            maiorId = ultimasDeteccoes[j].idDeteccao;
                                        }
                                    }

                                    deteccoes[i].idDeteccao = maiorId + 1;

                                    qtdFramesPrevisao += 1;


                                    if (qtdFramesPrevisao == qtdFramesPrevisaoTotal) {
                                        tempoPreverProximo = tempoPreverProximoLimite;
                                        qtdFramesPrevisao = 0;
                                    }
                                }

                            }
                            else {
                                //##########################
                                //######IMPLEMENTAR#########
                                //##########################                                
                            }
                        }
                        else {
                            //##########################
                            //######IMPLEMENTAR#########
                            //##########################
                        }

                    }
                }


                imshow("Câmera", frame);
                if (waitKey(5) >= 0)
                    break;
            }

        }
        system("cls");


    }

    return 0;
}
