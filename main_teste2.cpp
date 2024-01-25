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
    cv::Mat rosto;
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





//Funções
void criarDiretorio(String diretorio);
int adicionaFuncionarioFile(String arquivoGravacao);
vector<Deteccao> deteccaoSSD(dnn::Net network, Mat frame, int tamanho, float confiancaMinima);
bool validaDeteccao(Deteccao deteccao, float confiancaMinima, int larguraIdeal, int alturaIdeal);
vector<Pessoa> comparaPessoasComDeteccao(vector<Pessoa> listaDePessoas, Deteccao deteccao);
struct Pessoa getPessoaMaisProxima(vector<Pessoa>listaDePessoas);
void mostrarPessoaDetectada(Pessoa pessoaMaisProxima, Deteccao deteccao);
struct Deteccao getDeteccaoMaisProxima(vector<Deteccao>listaDedeteccoes);


//Variáveis globais
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

                        //calcula a distancia da detecção atual com todas as detecções globais
                        //dgi = deteccoes globais index
                        for (int dgi = 0; dgi < deteccoesGlobais.size(); dgi++) {

                            float x1 = deteccoesGlobais[dgi].box.startX;
                            float x2 = deteccoes[i].box.startX;

                            float y1 = deteccoesGlobais[dgi].box.startY;
                            float y2 = deteccoes[i].box.startY;

                            deteccoesGlobais[dgi].distancia = sqrt((pow(x1 - x2, 2) + (pow(y1 - y2, 2))));                            
                        }

                        //calcula a deteção com a menor distância
                        //dgi = deteccoes globais index
                        struct Deteccao deteccaoMaisProxima = {};
                        for (int dgi = 0; dgi < deteccoesGlobais.size(); dgi++) {

                            if (dgi == 0) {
                                deteccaoMaisProxima = deteccoesGlobais[dgi];
                            }
                            else {
                                if (deteccaoMaisProxima.distancia<deteccoesGlobais[dgi].distancia) {
                                    deteccaoMaisProxima = deteccoesGlobais[dgi];
                                }
                            }

                        }

                        //Verifica se realmente a deteção mais próxima está em uma proximidade aceitável.
                        //Caso não esteja a deteção é então acoplada a lista deteccaoMaisProxima
                        if (deteccaoMaisProxima.distancia < 20) {
                            
                            for (int dgi = 0; dgi < deteccoesGlobais.size(); dgi++) {

                                if (deteccoesGlobais[dgi].idDeteccao == deteccaoMaisProxima.idDeteccao) {
                                    
                                    deteccoesGlobais[dgi].deteccoes.push_back(deteccaoMaisProxima.rosto);
                                }
                            }                            
                        }
                        else {
                            
                            deteccaoMaisProxima.idDeteccao = deteccoesGlobais[deteccoesGlobais.size() - 1].idDeteccao + 1;
                            deteccoesGlobais.push_back(deteccaoMaisProxima);
                        }

                    }
                    else {

                        deteccoes[i].idDeteccao = 0;
                        deteccoesGlobais.push_back(deteccoes[i]);
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