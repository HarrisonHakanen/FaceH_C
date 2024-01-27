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



struct Recorrencia {
    std::vector<int> classificacoes;
    std::vector<int> qtdVezes;
};

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
    struct Pessoa pessoa;
    float distancia;
    int qtdFrames;
    int esperando;
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
void mostrarPessoaDetectada(Pessoa pessoaMaisProxima, Deteccao deteccao);
vector<Deteccao> calculaDistanciaDeteccaoParaDeteccoesGlobais(vector<Deteccao> deteccoesGlobais, Deteccao deteccao);
Deteccao getDeteccaoMaisProxima(vector<Deteccao> deteccoesGlobais);
vector<Deteccao> validaDeteccaoMaisProxima(vector<Deteccao>deteccoesGlobais, Deteccao deteccaoMaisProxima, Deteccao deteccaoAtual);
vector<int> retornaPrevisoes(Deteccao deteccaoGlobal);
Recorrencia getRecorrenciasDeClassificacao(vector<int>previsoes);
int getIndexDaClassificacaoMaisRecorrente(Recorrencia recorrencia);
Pessoa getPessoaPelaClassificacao(Recorrencia recorrencia, int indexMaior);
void escreveNome(Deteccao deteccao, InputOutputArray frame);


//Variáveis globais
const int tempoDeCadastro = 10;
const int opcao = 0;
const std::string nome = "";
const bool continuaGravacao = true;
const int fps = 60;
const int larguraIdeal = 130;
const int alturaIdeal = 180;
const float confiancaMinimaDeteccao = 0.9f;
const float confiancaMinimaDaPrevisao = 0.8;
const int qtdDeDeteccoesPorCluster = 15;




//Pastas
const std::string modeloSSDPath = "modelo_ssd";
const std::string pessoasCadastradasPath = "pessoas_cadastradas";
const std::string facesPath = "faces";
const std::string modelosYmlPath = "modelos";


//Modelos
const std::string arquivo_modelo = "modelo_ssd\\res10_300x300_ssd_iter_140000.caffemodel";
const std::string arquivo_prototxt = "modelo_ssd\\deploy.prototxt.txt";
const cv::dnn::Net network = cv::dnn::readNetFromCaffe(arquivo_prototxt, arquivo_modelo);


//Caminhos
const std::string arquivoGravacao = "pessoas_cadastradas\\pessoas.txt";
const std::string arquivoId = "pessoas_cadastradas\\ultimoId.txt";
const std::string modeloYml = modelosYmlPath + "\\" + "modelo_LBPH.yml";


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


            while (true) {

                cap.read(frame);

                vector<Deteccao> deteccoes = deteccaoSSD(network, frame, 300, 0.9f);                                
                

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
                            if (deteccoesGlobais[dgi].deteccoes.size() >= qtdDeDeteccoesPorCluster) {


                                //Realiza as previsões para todos os rostos detectados para a detecção global vigente
                                vector<int> previsoes = retornaPrevisoes(deteccoesGlobais[dgi]);
                                                              
                                //Separa as classificações pela quantidade de vezes que elas aparecem
                                Recorrencia recorrencia = getRecorrenciasDeClassificacao(previsoes);
                                
                                //Pega qual a classificação que mais recorrente
                                int indexMaior = getIndexDaClassificacaoMaisRecorrente(recorrencia);

                                //Pega a pessoa na lista de pessoas cadastradas através do index/id dela                               
                                deteccoesGlobais[dgi].pessoa = getPessoaPelaClassificacao(recorrencia, indexMaior);

                                //Escreve o nome da pessoa em cima da detecção dela
                                escreveNome(deteccoesGlobais[dgi],frame);                               

                            }
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

            arquivoCadastro << "Id:" << idFormulario << "|" << "Nome:" << nome << "\n";
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

            //O rosto em escala de cinza é a primeira deteção do objeto Deteccao
            //Ou seja, sempre que quando uma detecção existir, ela já vai ter por padrão um rosto nas detecções.
            vector<cv::Mat> deteccoes;
            deteccoes.push_back(roiGrey);

            Pessoa pessoa = {-1,"",box,0};

            struct Deteccao detec = { idDetec,frame,confianca,box,previsoes,deteccoes,pessoa,-1.0,0,false};


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


vector<Deteccao> validaDeteccaoMaisProxima(vector<Deteccao>deteccoesGlobais,Deteccao deteccaoMaisProxima,Deteccao deteccaoAtual) {

    if (deteccaoMaisProxima.distancia < 20 && deteccaoMaisProxima.distancia != -1) {

        for (int dgi = 0; dgi < deteccoesGlobais.size(); dgi++) {

            if (deteccoesGlobais[dgi].idDeteccao == deteccaoMaisProxima.idDeteccao) {

                //Atribui o rosto a detecção que esta na lista de deteccoesGlobais
                //E também é atualizado a posição atual do rosto
                deteccoesGlobais[dgi].deteccoes.push_back(deteccaoMaisProxima.deteccoes[0]);                
                deteccoesGlobais[dgi].box = deteccaoAtual.box;

            }
        }
    }
    else {

        deteccaoMaisProxima.idDeteccao = deteccoesGlobais[deteccoesGlobais.size() - 1].idDeteccao + 1;
        deteccoesGlobais.push_back(deteccaoMaisProxima);
    }

    return deteccoesGlobais;

}


vector<int> retornaPrevisoes(Deteccao deteccaoGlobal) {

    vector<int> previsoes;
    
    Ptr <face::FaceRecognizer> lbphClassifier = face::LBPHFaceRecognizer::create();


    if (filesystem::exists(modeloYml)) {

        lbphClassifier->read(modeloYml);

        for (int rostoIndex = 0; rostoIndex < deteccaoGlobal.deteccoes.size(); rostoIndex++) {
                        
            int prev = lbphClassifier->predict(deteccaoGlobal.deteccoes[rostoIndex]);

            previsoes.push_back(prev);            
        }

    }

    return previsoes;
}


Recorrencia getRecorrenciasDeClassificacao(vector<int>previsoes){

    vector<int> classificacoes;
    vector<int> qtdVezes;


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
    Pessoa pessoaEncontrada = {-1,"",box,-1};

    if (confiancaClassificacao > confiancaMinimaDaPrevisao) {

        if (filesystem::exists(arquivoGravacao)) {

            ifstream arquivo(arquivoGravacao);

            if (arquivo.is_open()) {

                string linha;


                int idPessoa = -1;
                string nome = "";
                bool encontrouPessoa = false;

                while (getline(arquivo, linha)) {

                    istringstream ss(linha);

                    string parte;


                    if (!encontrouPessoa) {

                        while (getline(ss, parte, '|')) {

                            //Pega o Id da pessoa
                            if (parte.find("Id") != string::npos) {

                                istringstream idParte(parte);
                                while (getline(idParte, parte, ':')) {

                                    if (parte != "Id") {

                                        if (maiorClassificacao == stoi(parte)) {
                                            idPessoa = stoi(parte);
                                            encontrouPessoa = true;
                                        }

                                        cout << parte;
                                    }
                                }
                            }

                            if (encontrouPessoa) {

                                //Pega o Nome da pessoa caso tenha encontrado ela pelo id
                                if (parte.find("Nome") != string::npos) {

                                    istringstream idParte(parte);
                                    while (getline(idParte, parte, ':')) {

                                        if (parte != "Nome") {

                                            nome = parte;
                                            cout << parte;
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


void escreveNome(Deteccao deteccao, InputOutputArray frame) {

    if (deteccao.pessoa.id != -1) {        

        cv::putText(
            frame,
            deteccao.pessoa.nome,
            Point(deteccao.box.startX, deteccao.box.startY),
            cv::FONT_HERSHEY_DUPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            2,
            false);
    }
}