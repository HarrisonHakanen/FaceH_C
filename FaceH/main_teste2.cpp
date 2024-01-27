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



struct Box {
    int startX;
    int startY;
    int endX;
    int endY;
};

struct Pessoa {
    int id;
    std::string nome;
    struct Box box;
    float distancia;
};


struct Previsao {
    int idPrevisao;
    std::vector<float>previsao;
};

struct Deteccao {
    int idDeteccao;
    cv::Mat imagem;
    float confiancaRetorno;
    struct Box box;
    std::vector<Previsao>previsoes;
    std::vector<cv::Mat> deteccoes;
    float distancia;
    int qtdFrames;
};


using std::cout;
using std::cin;
using std::endl;
using std::array;
using std::vector;
using std::getline;
using std::string;
using namespace cv;
using namespace std;


//Fun��es
void criarDiretorio(String diretorio);
int adicionaFuncionarioFile(String arquivoGravacao);
vector<Deteccao> deteccaoSSD(dnn::Net network, Mat frame, int tamanho, float confiancaMinima);
bool validaDeteccao(Deteccao deteccao, float confiancaMinima, int larguraIdeal, int alturaIdeal);
void mostrarPessoaDetectada(Pessoa pessoaMaisProxima, Deteccao deteccao);
vector<Deteccao> calculaDistanciaDeteccaoParaDeteccoesGlobais(vector<Deteccao> deteccoesGlobais, Deteccao deteccao);
Deteccao getDeteccaoMaisProxima(vector<Deteccao> deteccoesGlobais);


//Vari�veis globais
extern const int tempoDeCadastro = 10;
extern const int opcao = 0;
extern const std::string nome = "";
extern const bool continuaGravacao = true;
extern const int fps = 60;
extern const int larguraIdeal = 130;
extern const int alturaIdeal = 180;
extern const float confiancaMinimaDeteccao = 0.9f;


//Pastas
extern const std::string modeloSSDPath = "modelo_ssd";
extern const std::string pessoasCadastradasPath = "pessoas_cadastradas";
extern const std::string facesPath = "faces";
extern const std::string modelosYmlPath = "modelos";


//Modelos
extern const std::string arquivo_modelo = "modelo_ssd\\res10_300x300_ssd_iter_140000.caffemodel";
extern const std::string arquivo_prototxt = "modelo_ssd\\deploy.prototxt.txt";
extern const cv::dnn::Net network = cv::dnn::readNetFromCaffe(arquivo_prototxt, arquivo_modelo);


//Caminhos
extern const std::string arquivoGravacao = "pessoas_cadastradas\\pessoas.txt";
extern const std::string arquivoId = "pessoas_cadastradas\\ultimoId.txt";
extern const std::string modeloYml = modelosYmlPath + "\\" + "modelo_LBPH.yml";


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


            while (true) {

                cap.read(frame);

                vector<Deteccao> deteccoes = deteccaoSSD(network, frame, 300, 0.9f);                                
                

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
                        struct Deteccao deteccaoMaisProxima = getDeteccaoMaisProxima(deteccoesGlobais);



                        //Verifica se realmente a dete��o mais pr�xima est� em uma proximidade aceit�vel.
                        //Caso n�o esteja a dete��o � ent�o acoplada a lista deteccaoMaisProxima
                        if (deteccaoMaisProxima.distancia < 20 && deteccaoMaisProxima.distancia != -1) {
                            
                            for (int dgi = 0; dgi < deteccoesGlobais.size(); dgi++) {

                                if (deteccoesGlobais[dgi].idDeteccao == deteccaoMaisProxima.idDeteccao) {
                                    
                                    //Atribui o rosto a detec��o que esta na lista de deteccoesGlobais
                                    //E tamb�m � atualizado a posi��o atual do rosto
                                    deteccoesGlobais[dgi].deteccoes.push_back(deteccaoMaisProxima.deteccoes[0]);
                                    deteccoesGlobais[dgi].box = deteccoes[i].box;
                                }
                            }                            
                        }
                        else {
                            
                            deteccaoMaisProxima.idDeteccao = deteccoesGlobais[deteccoesGlobais.size() - 1].idDeteccao + 1;
                            deteccoesGlobais.push_back(deteccaoMaisProxima);
                        }



                        //################################################################
                        //########                                                ########
                        //########   PARTE 2 - REALIZANDO AS PRIMEIRAS PREVIS�ES  ########
                        //########                                                ########
                        //################################################################


                        for(int dgi = 0; dgi < deteccoesGlobais.size();dgi++) {

                            //Quando o objeto detec��o atinge 15 imagens do rosto, ent�o � realizado o algoritmo para
                            //fazer a previs�o de todos as 15 imagens e definir qual � a pessoa prevista para aquela detec��o.
                            if (deteccoesGlobais[dgi].deteccoes.size() >= 15) {

                                vector<int> previsoes;

                                //Realiza as previs�es para todos os rostos detectados para essa detec��o
                                for (int rostoIndex = 0; rostoIndex < deteccoesGlobais[dgi].deteccoes.size();rostoIndex++) {

                                    Ptr <face::FaceRecognizer> lbphClassifier = face::LBPHFaceRecognizer::create();


                                    if (filesystem::exists(modeloYml)) {

                                        lbphClassifier->read(modeloYml);
                                        int prev = lbphClassifier->predict(deteccoesGlobais[dgi].deteccoes[rostoIndex]);

                                        previsoes.push_back(prev);
                                    }
                                }


                                //Separa as classifica��es pela quantidade de vezes que elas aparecem                                
                                vector<int> classificacoes;
                                vector<int> qtdVezes;

                                
                                for (int prevIndex = 0; prevIndex < previsoes.size();prevIndex++) {
                                    
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

                                //Pega qual a classifica��o que mais aparece
                                int indexMaior = 0;
                                int maior = 0;
                                for (int qtdIndex = 0; qtdIndex < qtdVezes.size();qtdIndex++) {
                                    if (qtdIndex == 0) {
                                        maior = qtdVezes[qtdIndex];
                                        indexMaior = qtdIndex;
                                    }
                                    else {
                                        if (qtdVezes[qtdIndex] > maior) {
                                            maior = qtdVezes[qtdIndex];
                                            indexMaior = qtdIndex;
                                        }
                                    }
                                }

                                int maiorClassificacao = classificacoes[indexMaior];
                                int confiancaClassificacao = qtdVezes[indexMaior] / 15;

                            }
                        }

                        

                    }
                    else {

                        deteccoes[i].idDeteccao = 0;
                        deteccoesGlobais.push_back(deteccoes[i]);
                    }
                }


                imshow("C�mera", frame);
                if (waitKey(5) >= 0)
                    break;
            }

        }
        system("cls");


    }

    return 0;
}





void criarDiretorio(String diretorio) {

    if (!filesystem::exists(diretorio)) {

        if (filesystem::create_directories(diretorio)) {
            printf("O diret�rio %s foi criado com �xito.", diretorio.c_str());
        }
        else {
            printf("O diret�rio %s j� existe.", diretorio.c_str());
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

        // Verifique se a confian�a atende ao limiar
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

            //O rosto em escala de cinza � a primeira dete��o do objeto Deteccao
            //Ou seja, sempre que quando uma detec��o existir, ela j� vai ter por padr�o um rosto nas detec��es.
            vector<cv::Mat> deteccoes;
            deteccoes.push_back(roiGrey);

            struct Deteccao detec = { idDetec,frame,confianca,box,previsoes,deteccoes,-1.0,0 };


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


void mostrarPessoaDetectada(Pessoa pessoaMaisProxima, Deteccao deteccao) {

    if (pessoaMaisProxima.distancia < 20 && pessoaMaisProxima.distancia != -1) {

        HersheyFonts font = FONT_HERSHEY_COMPLEX;

        double fontScale = 0.5;

        Point org = Point(pessoaMaisProxima.box.startX, pessoaMaisProxima.box.startY - 10);

        pessoaMaisProxima.box = deteccao.box;


        putText(deteccao.imagem, pessoaMaisProxima.nome, org, font, fontScale, CV_RGB(118, 185, 0), 1, 8, false);
    }

}



vector<Deteccao> calculaDistanciaDeteccaoParaDeteccoesGlobais(vector<Deteccao> deteccoesGlobais,Deteccao deteccao) {

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

Deteccao getDeteccaoMaisProxima(vector<Deteccao> deteccoesGlobais) {
   
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