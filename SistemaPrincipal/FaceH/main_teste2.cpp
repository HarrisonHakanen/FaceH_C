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
        printf("Selecione uma op��o:\n1 - Adicionar pessoa\n2 - Listar pessoas cadastradas\n3 - Continuar sistema\n4 - Sair do sistema\n\n");

        cin >> escolha;

        if (escolha == 1) {


            system("cls");
            printf("Adicionar pessoa\n");

            int ultimoId = adicionaFuncionarioFile(arquivoGravacao);


            printf("Preparando c�mera para reconhecimento...");

            bool continuaGravacao = true;
            vector<int>idList;
            vector<Mat> faceList;

            Mat frame;

            VideoCapture cap;

            int deviceID = 0;
            int apiID = CAP_ANY;

            cap.open(deviceID, apiID);

            if (!cap.isOpened()) {
                cerr << "ERROR! Camera inacess�vel\n";
                return -1;
            }


            while (continuaGravacao)
            {

                cap.read(frame);

                vector<Deteccao> deteccoes = deteccaoSSD(network, frame, 300, 0.9f);

                //Inicio fun��o

                for (int i = 0; i < deteccoes.size(); i++) {

                    if (validaDeteccao(deteccoes[i], confiancaMinimaDeteccao, larguraIdeal, alturaIdeal)) {


                        imshow("C�mera", deteccoes[i].imagem);

                        idList.push_back(ultimoId);
                        faceList.push_back(deteccoes[i].deteccoes[0]);

                        qtdFramesCadastro += 1;


                        if (qtdFramesCadastro > qtdFramesCadastroTotal) {
                            continuaGravacao = false;
                            qtdFramesCadastro = 0;
                        }
                    }
                    else {
                        imshow("C�mera", frame);
                    }


                }
                imshow("C�mera", frame);

                //Fim fun��o

                if (waitKey(5) >= 0)
                    break;

            }

            cap.release();
            destroyAllWindows();


            //Inicio fun��o            
            Ptr <face::FaceRecognizer> lbphClassifier = face::LBPHFaceRecognizer::create();


            if (filesystem::exists(modeloYml)) {

                lbphClassifier->read(modeloYml);
                lbphClassifier->update(faceList, idList);

            }
            else {

                lbphClassifier->train(faceList, idList);

            }

            lbphClassifier->write(modeloYml);
            //Fim fun��o

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
                        //########   PARTE 1 - MANIPULANDO DETEC��ES  ########
                        //########                                    ########
                        //####################################################
                        

                        //calcula a distancia da detec��o atual com todas as detec��es globais                        
                        deteccoesGlobais = calculaDistanciaDeteccaoParaDeteccoesGlobais(deteccoesGlobais, deteccoes[i]);                        


                        //calcula a dete��o com a menor dist�ncia
                        //dgi = deteccoes globais index   
                        Deteccao deteccaoMaisProxima = getDeteccaoMaisProxima(deteccoesGlobais);



                        //Verifica se realmente a dete��o mais pr�xima est� em uma proximidade aceit�vel.
                        //Caso n�o esteja a dete��o � ent�o acoplada a lista deteccaoMaisProxima
                        deteccoesGlobais = validaDeteccaoMaisProxima(deteccoesGlobais,deteccaoMaisProxima,deteccoes[i]);                        
                        

                        //################################################################
                        //########                                                ########
                        //########   PARTE 2 - REALIZANDO AS PRIMEIRAS PREVIS�ES  ########
                        //########                                                ########
                        //################################################################


                        for(int dgi = 0; dgi < deteccoesGlobais.size();dgi++) {

                            //Quando o objeto detec��o atinge 15 imagens do rosto, ent�o � realizado o algoritmo para
                            //fazer a previs�o de todos as 15 imagens e definir qual � a pessoa prevista para aquela detec��o.                            
                            if (deteccoesGlobais[dgi].deteccoes.size() == qtdDeDeteccoesPorCluster) {


                                


                                //Realiza as previs�es para todos os rostos detectados para a detec��o global vigente
                                vector<int> previsoes = retornaPrevisoes(deteccoesGlobais[dgi]);

                                //Separa as classifica��es pela quantidade de vezes que elas aparecem
                                Recorrencia recorrencia = getRecorrenciasDeClassificacao(previsoes);

                                //Pega qual a classifica��o que mais recorrente
                                int indexMaior = getIndexDaClassificacaoMaisRecorrente(recorrencia);



                                cout << "Index: " << indexMaior;
                                //Pega a pessoa na lista de pessoas cadastradas atrav�s do index/id dela                               
                                deteccoesGlobais[dgi].pessoa = getPessoaPelaClassificacao(recorrencia, indexMaior);


                                //Com exce��o do ultimo rosto detectado, os demais s�o todos removidos do vetor de dete��es do objeto Deteccao
                                cv::Mat ultimoRosto = deteccoesGlobais[dgi].deteccoes[deteccoesGlobais[dgi].deteccoes.size() - 1];

                                deteccoesGlobais[dgi].deteccoes.clear();                                

                                deteccoesGlobais[dgi].deteccoes.push_back(ultimoRosto);
                                
                                deteccoesGlobais[dgi].esperando = true;
                                deteccoesGlobais[dgi].tempoEspera = hiatoDeClusterizacao;                                

                            }
                            //Escreve o nome da pessoa em cima da detec��o dela
                            escreveNome(deteccoesGlobais[dgi],deteccoes[i], frame);
                            
                        }                        
                    }
                    else {

                        deteccoes[i].idDeteccao = 0;
                        deteccoesGlobais.push_back(deteccoes[i]);
                    }
                }


                cv::imshow("C�mera", frame);
                if (waitKey(5) >= 0)
                    break;

                ciclo += 1;
            }

        }
        system("cls");


    }

    return 0;
}



