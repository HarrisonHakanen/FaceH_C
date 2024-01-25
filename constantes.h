#ifndef CONSTANTES_H
#define CONSTANTES_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>



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

#endif // CONSTANTES_H


