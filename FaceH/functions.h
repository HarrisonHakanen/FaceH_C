#ifndef PLAYER_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define PLAYER_H

#include <stdio.h>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>


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
    bool esperando;
    int tempoEspera;
};


void criarDiretorio(std::string diretorio);
int adicionaFuncionarioFile(std::string arquivoGravacao);
std::vector<Deteccao> deteccaoSSD(cv::dnn::Net network, cv::Mat frame, int tamanho, float confiancaMinima);
bool validaDeteccao(Deteccao deteccao, float confiancaMinima, int larguraIdeal, int alturaIdeal);
void mostrarPessoaDetectada(Pessoa pessoaMaisProxima, Deteccao deteccao);
std::vector<Deteccao> calculaDistanciaDeteccaoParaDeteccoesGlobais(std::vector<Deteccao> deteccoesGlobais, Deteccao deteccao);
Deteccao getDeteccaoMaisProxima(std::vector<Deteccao> deteccoesGlobais);
std::vector<Deteccao> validaDeteccaoMaisProxima(std::vector<Deteccao>deteccoesGlobais, Deteccao deteccaoMaisProxima, Deteccao deteccaoAtual);
std::vector<int> retornaPrevisoes(Deteccao deteccaoGlobal);
Recorrencia getRecorrenciasDeClassificacao(std::vector<int>previsoes);
int getIndexDaClassificacaoMaisRecorrente(Recorrencia recorrencia);
Pessoa getPessoaPelaClassificacao(Recorrencia recorrencia, int indexMaior);
void escreveNome(Deteccao deteccao, Deteccao deteccaoSSD, cv::InputOutputArray frame);

#endif