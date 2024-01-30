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

#include "global.h"
#include "functions.h"



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


            while (continuaGravacao)
            {

                cap.read(frame);

                vector<Deteccao> deteccoes = deteccaoSSD(network, frame, 300, 0.9f);

                //Inicio função

                for (int i = 0; i < deteccoes.size(); i++) {

                    if (validaDeteccao(deteccoes[i], confiancaMinimaDeteccao, larguraIdeal, alturaIdeal)) {


                        imshow("Câmera", deteccoes[i].imagem);

                        idList.push_back(ultimoId);
                        faceList.push_back(deteccoes[i].deteccoes[0]);

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


            vector<Deteccao>deteccoesGlobais;


            Mat frame;

            VideoCapture cap;

            int deviceID = 0;
            int apiID = CAP_ANY;

            cap.open(deviceID, apiID);


            bool entrouNaVerificacao = false;
            int ciclo = 0;


            while (true) {
                

                cap.read(frame);

                vector<Deteccao> deteccoes = deteccaoSSD(network, frame, 300, 0.9f);                                
               

                if (ciclo >= 50) {

                    cout << "\n" << deteccoesGlobais.size();
                    entrouNaVerificacao = true;

                }             


                for (int i = 0; i < deteccoes.size(); i++) {

                    if (deteccoesGlobais.size() > 0) {
                        

                        //####################################################
                        //########                                    ########
                        //########   PARTE 1 - MANIPULANDO DETECÇÕES  ########
                        //########                                    ########
                        //####################################################
                        

                        //calcula a distancia da detecção atual com todas as detecções globais                        
                        deteccoesGlobais = calculaDistanciaDeteccaoParaDeteccoesGlobais(deteccoesGlobais, deteccoes[i]);                        


                        //calcula a deteção com a menor distância
                        //dgi = deteccoes globais index   
                        Deteccao deteccaoMaisProxima = getDeteccaoMaisProxima(deteccoesGlobais);



                        //Verifica se realmente a deteção mais próxima está em uma proximidade aceitável.
                        //Caso não esteja a deteção é então acoplada a lista deteccaoMaisProxima
                        deteccoesGlobais = validaDeteccaoMaisProxima(deteccoesGlobais,deteccaoMaisProxima,deteccoes[i]);                        
                        

                        //################################################################
                        //########                                                ########
                        //########   PARTE 2 - REALIZANDO AS PRIMEIRAS PREVISÕES  ########
                        //########                                                ########
                        //################################################################


                        for(int dgi = 0; dgi < deteccoesGlobais.size();dgi++) {

                            //Quando o objeto detecção atinge 15 imagens do rosto, então é realizado o algoritmo para
                            //fazer a previsão de todos as 15 imagens e definir qual é a pessoa prevista para aquela detecção.                            
                            if (deteccoesGlobais[dgi].deteccoes.size() == qtdDeDeteccoesPorCluster) {


                                


                                //Realiza as previsões para todos os rostos detectados para a detecção global vigente
                                vector<int> previsoes = retornaPrevisoes(deteccoesGlobais[dgi]);

                                //Separa as classificações pela quantidade de vezes que elas aparecem
                                Recorrencia recorrencia = getRecorrenciasDeClassificacao(previsoes);

                                //Pega qual a classificação que mais recorrente
                                int indexMaior = getIndexDaClassificacaoMaisRecorrente(recorrencia);



                                cout << "Index: " << indexMaior;
                                //Pega a pessoa na lista de pessoas cadastradas através do index/id dela                               
                                deteccoesGlobais[dgi].pessoa = getPessoaPelaClassificacao(recorrencia, indexMaior);


                                //Com exceção do ultimo rosto detectado, os demais são todos removidos do vetor de deteções do objeto Deteccao
                                cv::Mat ultimoRosto = deteccoesGlobais[dgi].deteccoes[deteccoesGlobais[dgi].deteccoes.size() - 1];

                                deteccoesGlobais[dgi].deteccoes.clear();                                

                                deteccoesGlobais[dgi].deteccoes.push_back(ultimoRosto);
                                
                                deteccoesGlobais[dgi].esperando = true;
                                deteccoesGlobais[dgi].tempoEspera = hiatoDeClusterizacao;                                

                            }
                            //Escreve o nome da pessoa em cima da detecção dela
                            escreveNome(deteccoesGlobais[dgi],deteccoes[i], frame);
                            
                        }                        
                    }
                    else {

                        deteccoes[i].idDeteccao = 0;
                        deteccoesGlobais.push_back(deteccoes[i]);
                    }
                }


                cv::imshow("Câmera", frame);
                if (waitKey(5) >= 0)
                    break;

                ciclo += 1;
            }

        }
        system("cls");


    }

    return 0;
}



