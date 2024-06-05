#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/serialize.h>
#include <dlib/matrix.h>
#include <dlib/opencv/cv_image.h>
#include <filesystem>

#include "functions.h"
#include "global.h"



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
            

            if (validaTamanhoDeteccao(startX,startY,endX,endY,frame)) {

                cv::Mat roi = frame(cv::Range(startY, endY), cv::Range(startX, endX));

                cv::Mat roiResized;
                cv::resize(roi, roiResized, cv::Size(60, 80), 0, 0, cv::INTER_LINEAR);

                cv::Mat roiGrey;
                cv::cvtColor(roiResized, roiGrey, cv::COLOR_BGR2GRAY);


                //O rosto em escala de cinza é a primeira deteção do objeto Deteccao
                //Ou seja, sempre que quando uma detecção existir, ela já vai ter por padrão um rosto nas detecções.
                std::vector<cv::Mat> deteccoes;
                deteccoes.push_back(roiGrey);


                struct Deteccao detec = { idDetec,frame,confianca,box,deteccoes,-1.0 };

                idDetec += 1;

                deteccoesPessoa.push_back(detec);
            }            
        }
    }   

    return deteccoesPessoa;
}


bool validaTamanhoDeteccao(int startX, int startY, int endX, int endY, cv::Mat frame) {

    bool retorno = false;
    if ((startY > 0 && endY > 0 && startX > 0 && endX > 0)) {

        if (startY < frame.rows && startY < frame.cols) {
            if (startY < frame.rows && startY < frame.cols) {
                if (endX < frame.rows && endX < frame.cols) {
                    if (endY < frame.rows && endY < frame.cols) {
                        retorno = true;
                    }
                }
            }
        }
    }
    return retorno;
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


ObjTreinamento populaListasTreinamento(cv::dnn::Net network, ObjTreinamento treinamento, std::string caminhoDaPastaStr, std::string arquivoStr, std::string parte) {

    std::string caminhoArquivo = caminhoDaPastaStr + "\\" + arquivoStr;

    cv::Mat grayscale_image;
    std::vector<cv::Rect> features;

    cv::Mat imagem = cv::imread(caminhoArquivo);

    cvtColor(imagem, grayscale_image, cv::COLOR_BGR2GRAY);
    equalizeHist(grayscale_image, grayscale_image);

    //haarcascade.detectMultiScale(grayscale_image, features, 1.1, 4, 0, Size(30, 30));

    /*
    std::vector<Deteccao> deteccoes = deteccaoSSD(network, imagem, 150, 0.9f);

    for (int i = 0; i < deteccoes.size(); i++) {

        if (validaDeteccao(deteccoes[i], confiancaMinimaDeteccao, larguraIdeal, alturaIdeal)) {

            treinamento.idList.push_back(stoi(parte));
            treinamento.faceList.push_back(deteccoes[i].deteccoes[0]);

        }
    }
    */

    return treinamento;
}


void realizaTreinamento(ObjTreinamento treinamento, std::string arquivoModelos, std::string arquivoTodosModelos, frontal_face_detector detector_face, shape_predictor detector_pontos, anet_type descritor) {
    
    cv::Ptr <cv::face::FaceRecognizer> lbphClassifier = cv::face::LBPHFaceRecognizer::create();

    //Cria o diretório para o modelo da pessoa
    std::string modeloDiretorio = modelosYmlPath + "\\" + "Pessoa_" + std::to_string(treinamento.idList[0]);
    std::string dadosPessoais = modeloDiretorio + "\\" + "dados.txt";
    criarDiretorio(modeloDiretorio);

    //Define o caminho do modelo LBPH da pessoa para o a pasta que foi criada
    std::string modeloYml = modeloDiretorio + "\\" + std::to_string(treinamento.idList[0]) + "_LBPH.yml";


    //Intera com a face da pessoa
    std::vector<matrix<rgb_pixel>> faces;
    for (int i = 0; i < treinamento.faceList.size(); i++) {

        cv::Mat imagem;
        cv::cvtColor(treinamento.faceList[i], imagem, cv::COLOR_GRAY2BGR);


        //Converte o Image do OpenCv para o Dlib
        dlib::array2d<bgr_pixel> dlibImg;
        dlib::assign_image(dlibImg, dlib::cv_image<bgr_pixel>(imagem));


        //Os rostos são convertidos em um formato que dê para ser lido pelo descritor
        for (auto face : detector_face(dlibImg)) {

            auto pontos = detector_pontos(dlibImg, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(dlibImg, get_face_chip_details(pontos, 150, 0.25), face_chip);
            faces.push_back(std::move(face_chip));

        }
    }

    //Extrai os descritores dos rostos e para cada rosto um arquivo txt onde contém os descritores é criado.
    std::vector<matrix<float, 0, 1>> face_descriptors = descritor(faces);

    for (int indexDesc = 0; indexDesc < face_descriptors.size(); indexDesc++) {

        std::string descritor = modeloDiretorio + "\\" + "descritor" + std::to_string(indexDesc) + ".txt";
        std::vector<float> descritoresValue;
                    
        for (long j = 0; j < face_descriptors[indexDesc].nr(); ++j) {               
            descritoresValue.push_back(face_descriptors[indexDesc](j));
        }        
        
        std::string descritorContent = "";        

        for (float valor : descritoresValue) {            
            descritorContent += to_string(valor) + "\n";
        }

        //Grava os descritores                
        escreverArquivo(descritor, descritorContent,false);
    }


    //Treina o modelo LBPH
    lbphClassifier->train(treinamento.faceList, treinamento.idList);
    lbphClassifier->write(modeloYml);


    //Grava o modelo no arquivo de modelos novos
    escreverArquivo(arquivoModelos, modeloDiretorio,false);

    //Grava o modelo no arquivo de todos os modelos
    escreverArquivo(arquivoTodosModelos, modeloDiretorio, false);
    
    escreverArquivo(dadosPessoais,"IdPessoa["+std::to_string(treinamento.idList[0]),false);
}


void minMaxNormalization(std::vector<float>& data, double min, double max) {

    double range = max - min;

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (data[i] - min) / range;

        // Ajuste para o mínimo ou máximo se estiver fora da escala
        if (data[i] < min)
            data[i] = min;
        else if (data[i] > max)
            data[i] = max;
    }
}


void escreverArquivo(std::string nomeArquivo, auto conteudo,boolean sobrescrever) {


    if (sobrescrever) {

        std::ofstream arquivoCadastro(nomeArquivo);
        arquivoCadastro << conteudo;
        arquivoCadastro.close();
    }
    else {

        if (std::filesystem::exists(nomeArquivo)) {

            std::ofstream arquivoCadastro(nomeArquivo, std::ios::app);

            if (arquivoCadastro.is_open()) {
                arquivoCadastro << "\n" << conteudo;
                arquivoCadastro.close();
            }

        }
        else {
            std::ofstream arquivoCadastro(nomeArquivo);
            arquivoCadastro << conteudo;
            arquivoCadastro.close();
        }
    }    
}


void criarDiretorio(std::string diretorio) {

    if (!std::filesystem::exists(diretorio)) {

        if (std::filesystem::create_directories(diretorio)) {
            cout<<"O diretório %s foi criado com êxito."<< diretorio.c_str()<<"\n";
        }
        else {
            cout<<"O diretório %s já existe."<< diretorio.c_str()<<"\n";
        }
    }
}

std::vector<Previsao> retornaPrevisoes(Deteccao deteccaoGlobal, std::string modeloYml) {

    std::vector<Previsao> previsoes;

    cv::Ptr <cv::face::FaceRecognizer> lbphClassifier = cv::face::LBPHFaceRecognizer::create();


    if (std::filesystem::exists(modeloYml)) {

        lbphClassifier->read(modeloYml);

        for (int rostoIndex = 0; rostoIndex < deteccaoGlobal.deteccoes.size(); rostoIndex++) {

            int predictedClass;
            double confidence;

            lbphClassifier->predict(deteccaoGlobal.deteccoes[rostoIndex], predictedClass, confidence);

            Previsao previsao = { predictedClass,confidence };

            previsoes.push_back(previsao);
        }
    }
    return previsoes;
}


Recorrencia getRecorrenciasDeClassificacao(std::vector<Previsao>previsoes) {

    std::vector<int> classificacoes;
    std::vector<int> qtdVezes;


    for (int prevIndex = 0; prevIndex < previsoes.size(); prevIndex++) {

        int qtd = 0;
        bool achou = false;
        bool verificou = false;
        for (int subIndex = prevIndex; subIndex < previsoes.size(); subIndex++) {

            if (!verificou) {

                for (int classIndex = 0; classIndex < classificacoes.size(); classIndex++) {

                    if (classificacoes[classIndex] == previsoes[subIndex].previsao) {
                        achou = true;
                    }
                }

                if (!achou) {

                    classificacoes.push_back(previsoes[subIndex].previsao);
                    qtd += 1;
                }

                verificou = true;
            }
            else {

                if (previsoes[prevIndex].previsao == previsoes[subIndex].previsao) {
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


std::vector<std::string>splitText(char splitChar, std::string texto) {

    std::vector<std::string> partes;
    std::string parte = "";

    for (char c : texto) {
        if (c == splitChar) {
            partes.push_back(parte);
            parte = "";
        }
        else {
            parte += c;
        }
    }

    partes.push_back(parte);
    return partes;
}

float normalize(float x, float min_val, float max_val) {
    return (x - min_val) / (max_val - min_val);
}




