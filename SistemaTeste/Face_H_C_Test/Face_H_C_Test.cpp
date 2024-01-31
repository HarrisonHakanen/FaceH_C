#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc/imgproc.hpp>



#include <iostream>
#include <stdio.h>
#include <filesystem>
#include <vector>
#include <string>


struct ObjTreinamento {

    std::vector<int> idList;
    std::vector<cv::Mat> faceList;
};


struct Box {
    int startX;
    int startY;
    int endX;
    int endY;
};

struct Deteccao {
    int idDeteccao;
    cv::Mat imagem;
    float confiancaRetorno;
    struct Box box;
    std::vector<cv::Mat> deteccoes;
    float distancia;
};

namespace fs = std::filesystem;



//Modelo SSD
std::string arquivo_modelo = "modelo_ssd\\res10_300x300_ssd_iter_140000.caffemodel";
std::string arquivo_prototxt = "modelo_ssd\\deploy.prototxt.txt";
cv::dnn::Net network = cv::dnn::readNetFromCaffe(arquivo_prototxt, arquivo_modelo);


//Pastas
std::string modelosYmlPath = "modelos";
std::string imagensTeste = "ImagensTeste";


//Funções
std::vector<Deteccao> deteccaoSSD(cv::dnn::Net network, cv::Mat frame, int tamanho, float confiancaMinima);
bool validaDeteccao(Deteccao deteccao, float confiancaMinima, int larguraIdeal, int alturaIdeal);
ObjTreinamento populaListasTreinamento(ObjTreinamento treinamento, std::string caminhoDaPastaStr, std::string arquivoStr, std::string parte);
void realizaTreinamento(ObjTreinamento treinamento);
void criarDiretorio(std::string diretorio);


//Variáveis
int larguraIdeal = 130;
int alturaIdeal = 180;
float confiancaMinimaDeteccao = 0.9f;



int main()
{

    //###############################################################################################
    //########                                                                               ########
    //########   COMO OS TREINAMENTOS JÁ FORAM FEITOS TODO O CÓDIGO A SEGUIR SERÁ COMENTADO  ########
    //########                                                                               ########
    //###############################################################################################
    /*
    criarDiretorio(modelosYmlPath);

    std::string caminhoDaPastaStr = "C:\\Users\\harri\\Documents\\Visual studio projects\\Face_H_C_Test\\Face_H_C_Test\\Face dataset";
    fs::path caminhoDaPasta = caminhoDaPastaStr;

    if (fs::exists(caminhoDaPasta) && fs::is_directory(caminhoDaPasta)) {


        std::vector<int> idList;
        std::vector<cv::Mat> faceList;

        ObjTreinamento treinamento = { idList,faceList };


        std::string indexAnterior = "";

        int i = 0;
        for (const auto& arquivo : fs::directory_iterator(caminhoDaPasta)) {


            std::string arquivoStr = arquivo.path().filename().string();
            std::istringstream arquivoStream(arquivoStr);

            std::string parte;


            int indexParte = 0;

            while (std::getline(arquivoStream, parte, '-')) {

                if (indexParte == 0) {

                    std::cout << parte << "-";

                    if (i == 0) {

                        indexAnterior = parte;

                        treinamento = populaListasTreinamento(treinamento, caminhoDaPastaStr, arquivoStr, parte);

                    }
                    else {
                        if (indexAnterior == parte) {

                            treinamento = populaListasTreinamento(treinamento, caminhoDaPastaStr, arquivoStr, parte);

                        }
                        else {


                            realizaTreinamento(treinamento);

                            treinamento.idList.clear();
                            treinamento.faceList.clear();
                            indexAnterior = parte;

                            treinamento = populaListasTreinamento(treinamento, caminhoDaPastaStr, arquivoStr, parte);
                        }
                    }
                }

                if (indexParte == 1) {
                    std::cout << parte << "\n";
                }

                indexParte += 1;
            }


            i += 1;
        }
    }
    else {
        std::cout << "Caminho inválido ou não é uma pasta." << std::endl;
    }

    */

    std::string caminhoTesteStr = "C:\\Users\\harri\\Documents\\Visual studio projects\\Face_H_C_Test\\Face_H_C_Test\\ImagensTeste";
    fs::path caminhoTeste = caminhoTesteStr;

    if (fs::exists(caminhoTeste) && fs::is_directory(caminhoTeste)) {



        for (const auto& arquivo : fs::directory_iterator(caminhoTeste)) {

            std::string arquivoStr = caminhoTesteStr +"\\" + arquivo.path().filename().string();

            cv::Mat imagem = cv::imread(arquivoStr);

            std::vector<Deteccao> deteccoes = deteccaoSSD(network, imagem, 200, 0.7f);

            for (int i = 0; i < deteccoes.size(); i++) {

                if (validaDeteccao(deteccoes[i], confiancaMinimaDeteccao, larguraIdeal, alturaIdeal)) {




                    cv::rectangle(imagem, cv::Point(deteccoes[i].box.startX, deteccoes[i].box.startY), cv::Point(deteccoes[i].box.endX, deteccoes[i].box.endY), cv::Scalar(0, 255, 0), 2);
                                                           
                }
            }

            imshow("Imagem", imagem);
            int k = cv::waitKey(0);

        }

    }



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
            

            //O rosto em escala de cinza é a primeira deteção do objeto Deteccao
            //Ou seja, sempre que quando uma detecção existir, ela já vai ter por padrão um rosto nas detecções.
            std::vector<cv::Mat> deteccoes;
            deteccoes.push_back(roiGrey);
          

            struct Deteccao detec = { idDetec,frame,confianca,box,deteccoes,-1.0};


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


ObjTreinamento populaListasTreinamento(ObjTreinamento treinamento,std::string caminhoDaPastaStr,std::string arquivoStr,std::string parte) {

    std::string caminhoArquivo = caminhoDaPastaStr + "\\" + arquivoStr;

    cv::Mat imagem = cv::imread(caminhoArquivo);

    std::vector<Deteccao> deteccoes = deteccaoSSD(network, imagem, 300, 0.9f);

    for (int i = 0; i < deteccoes.size(); i++) {

        if (validaDeteccao(deteccoes[i], confiancaMinimaDeteccao, larguraIdeal, alturaIdeal)) {

            treinamento.idList.push_back(stoi(parte));
            treinamento.faceList.push_back(deteccoes[i].deteccoes[0]);

        }
    }

    return treinamento;
}


void realizaTreinamento(ObjTreinamento treinamento) {

    cv::Ptr <cv::face::FaceRecognizer> lbphClassifier = cv::face::LBPHFaceRecognizer::create();

    std::string modeloYml = modelosYmlPath + "\\" + std::to_string(treinamento.idList[0]) + "_LBPH.yml";

    
    lbphClassifier->train(treinamento.faceList, treinamento.idList);
    //lbphClassifier->read(modeloYml);
    //lbphClassifier->update(treinamento.faceList, treinamento.idList);
    //id, confidence_face = lbphClassifier.predict(face)
    lbphClassifier->write(modeloYml);
    
}


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