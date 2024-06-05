#ifndef PLAYER_H  
#define PLAYER_H

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

using namespace dlib;



struct Box {
    int startX;
    int startY;
    int endX;
    int endY;
};

struct Recorrencia {
    std::vector<int> classificacoes;
    std::vector<int> qtdVezes;
};

struct ObjTreinamento {
    std::vector<string>nome;
    std::vector<int> idList;
    std::vector<cv::Mat> faceList;
    std::vector<string>caminhoImagens;
};


struct Deteccao {
    int idDeteccao;
    cv::Mat imagem;
    float confiancaRetorno;
    struct Box box;
    std::vector<cv::Mat> deteccoes;
    float distancia;
};
struct Previsao {
    int previsao;
    int confianca;
};



template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;
template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;
template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;
template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
    alevel0<
    alevel1<
    alevel2<
    alevel3<
    alevel4<
    max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
    input_rgb_image_sized<150>
    >>>>>>>>>>>>;


//Funções
std::vector<Deteccao> deteccaoSSD(cv::dnn::Net network, cv::Mat frame, int tamanho, float confiancaMinima);
bool validaDeteccao(Deteccao deteccao, float confiancaMinima, int larguraIdeal, int alturaIdeal);
ObjTreinamento populaListasTreinamento(cv::dnn::Net network, ObjTreinamento treinamento, std::string caminhoDaPastaStr, std::string arquivoStr, std::string parte);
void realizaTreinamento(ObjTreinamento treinamento, std::string arquivoModelos, std::string arquivoTodosModelos, frontal_face_detector detector_face, shape_predictor detector_pontos, anet_type descritor);
void criarDiretorio(std::string diretorio);
std::vector<Previsao> retornaPrevisoes(Deteccao deteccaoGlobal, std::string modeloYml);
Recorrencia getRecorrenciasDeClassificacao(std::vector<Previsao>previsoes);
int getIndexDaClassificacaoMaisRecorrente(Recorrencia recorrencia);
std::vector<std::string>splitText(char splitChar, std::string texto);
float normalize(float x, float min_val, float max_val);
void escreverArquivo(std::string nomeArquivo, auto conteudo, boolean sobrescrever);
void minMaxNormalization(std::vector<float>& data, double min, double max);
bool validaTamanhoDeteccao(int startX, int startY, int endX, int endY, cv::Mat frame);
#endif