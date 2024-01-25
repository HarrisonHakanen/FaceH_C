#ifndef MYSTRUCT_H
#define MYSTRUCT_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


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

#endif