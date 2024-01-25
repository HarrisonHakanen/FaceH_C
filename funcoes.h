#ifndef FUNCOES_H
#define FUNCOES_H

#include <opencv2/opencv.hpp>


//Funções
void criarDiretorio(std::string diretorio);
int adicionaFuncionarioFile(std::string arquivoGravacao);
std::vector<Deteccao> deteccaoSSD(cv::dnn::Net network, cv::Mat frame, int tamanho, float confiancaMinima);
bool validaDeteccao(Deteccao deteccao, float confiancaMinima, int larguraIdeal, int alturaIdeal);
std::vector<Pessoa> comparaPessoasComDeteccao(std::vector<Pessoa> listaDePessoas, Deteccao deteccao);
struct Pessoa getPessoaMaisProxima(std::vector<Pessoa>listaDePessoas);
void mostrarPessoaDetectada(Pessoa pessoaMaisProxima, Deteccao deteccao);
struct Deteccao getDeteccaoMaisProxima(std::vector<Deteccao>listaDedeteccoes);


#endif // FUNCOES_H