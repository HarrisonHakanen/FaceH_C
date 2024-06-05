#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/serialize.h>
#include <dlib/matrix.h>
#include <dlib/opencv/cv_image.h>

#include <direct.h>
#include <map>

#include "global.h"
#include "functions.h"
#include "global_kmeans.h"
#include "functions_kmeans.h"
#include "global_pca.h"
#include "functions_pca.h";
#include <cmath>
#include <chrono>
#include <omp.h>
#include <thread>
#include <atomic>


struct CentroidesRegistros {
    std::vector<Centroid> centroides;
    std::vector<Registro> registros;
};


struct FuncClust {
    int idFunc;
    float distancia;
};

struct ClustFunc {
    int idClust;
    std::vector<FuncClust> funcs;
};

struct ClustQuant {
    int idCluster;
    int quantidade;
};

struct RetPartition {
    int partition;
    std::vector<FuncClust> vetor;
};

struct FuncionarioEncontrado {
    int idFuncionario;    
    int idCentroide;
    double distancia;
    double similaridade;
};


std::mutex mtx;
std::atomic<bool> encontrouFuncionario(false);
std::atomic<FuncionarioEncontrado> funcFind;


namespace fs = std::filesystem;
using namespace dlib;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


std::vector<Centroid> carregarCentroides();
bool containString(std::string pesquisa, std::string texto);
std::vector<Registro> carregarDescritores(std::vector<string>todosOsCadastros, int quantFramesKmeansPCA);
std::vector<Registro> carregarDescritoresById(int idNovaPessoa, std::string tipoDescritor, std::vector<std::string> todosOsRegistros, int quantFramesKmeansPCA);
std::vector<std::vector<float>> treinarPca(std::vector<Registro> registrosCarregados);
std::string descritoresString(std::vector<float> matrizResultanteLinha);
int salvaDescritoresReduzidos(std::vector<std::string> listaDeCaminhos, int sizeListaCaminhos, int idRegistro, int contadorId, std::string descritorReduzidoContent, int ultimaPosicao);
std::string getIdPessoa(std::string dadosPessoais);
std::vector<Registro> carregarDescritoresReduzidos(std::vector<string>todosOsCadastros, int quantFramesKmeansPCA);
std::vector<Registro> kmeansFit(std::vector<Centroid>centroides, std::vector<Registro> registrosReduzidos);
void salvaCentroides(std::vector<Centroid> centroides);
std::vector<std::string> carregaPessoasNovas();
std::vector<std::string> carregarTodasPessoas();
void substituirValorArquivo(std::string nomeArquivo, auto conteudo);
std::vector<std::vector<float>> carregaAutoVetores();
void salvaClasseDeCadaCadastro(std::vector<Registro> registrosReduzidos);
std::vector<Registro> aplicaPcaEmNovoRegistro(std::vector<Registro>registrosCarregados, std::string pessoaNome);
void salvarClassificacoes(std::vector<Centroid>centroides, std::vector<Registro>todosOsRegistros);
std::vector<float> reduzDimensao(Registro registro);
int kmeansPredict(std::vector<Centroid>centroides, Registro registro);
std::vector<float> carregaArquivo(std::string arquivo);


bool salvarDescritores(ObjTreinamento treinamento, frontal_face_detector detector_face, shape_predictor detector_pontos, anet_type extrator_descritor_facial, string modeloDiretorio);
void realizarTreinamentoLBPH(ObjTreinamento treinamento, int quantFramesLBPH, string modeloDiretorio);
ObjTreinamento populaObjTreinamento(auto& pessoa, int quantFramesKmeansPCA, std::vector<std::string> todosOsCadastros);
bool salvaDescritores_treinaLBPH(std::string nomePasta, ObjTreinamento treinamento, frontal_face_detector detector_face, shape_predictor detector_pontos, anet_type extrator_descritor_facial, int quantFramesLBPH);
int getIdFromTxt(std::string pessoaPath);
std::vector<std::vector<float>> carregaDescritoresFromTxt(std::string pessoaPath, std::string tipoDescritor);
std::vector<ClustFunc> carregarCentroidesClassificacoes();
std::vector<FuncClust> swap(std::vector<FuncClust>arr, int pos1, int pos2);
void salvaMediasDescritoresReduzidos(std::vector<Registro> registrosClusters);


std::vector<double> retornaMedia(std::vector<std::vector<double>>vetores);
int confereTamanhoMatriz(std::vector<std::vector<float>> vetores);
float similaridaDeCosseno(std::vector<std::vector<float>> vetores, int tamanhoVetores);
float normDifference(const std::vector<float>vec1, const std::vector<float>vec2);
std::vector<FuncClust> swapVector(std::vector<FuncClust>vetor, int pos1, int pos2);
RetPartition partition(std::vector<FuncClust>arr, int low, int high);
std::vector<FuncClust> quickSort(std::vector<FuncClust>arr, int low, int high);
std::vector<Centroid>criarCentroides(std::vector<Registro> registrosReduzidos);


std::vector<ClustFunc> carregarCentroidesClassificacoesTeste(std::string path);
std::vector<Registro> kmeansFitTeste(std::vector<Centroid>centroides, std::vector<Registro> registrosReduzidos);
std::vector<std::string> carregarPessoasKmeans();
std::vector<Centroid> carregarCentroidesByIds(std::vector<int> idsCentroides, std::string path);
Centroid carregarCentroideById(int idCentroide);
void atualizarCentroides(std::vector<Centroid> centroidesAtualizados);
void atualizarCentroidesClassificacoes(std::vector<ClustFunc>classificacoes, std::string path);
Centroid redefinePosicaoCentroide(Centroid centroideCheio, Centroid centroideReposicionado);
void adicionarCentroides(Centroid centroideAtualizado);
CentroidesRegistros remodelarCentroides(std::vector <string> todosOsRegistros, int quantFramesKmeansPCA);
std::string fibonnaciSearch(std::vector<std::string> todosOsRegistros, int splitInicio, int splitFim, int idPesquisado);
void salvarClassificacoesPorRegistro(std::vector<Registro>registros, std::vector<string> todosOsRegistros);
Registro carregarMedias(int idNovaPessoa, std::string tipoDescritor, std::vector<std::string> todosOsRegistros);
void extrairDescritoresThread(matrix<rgb_pixel> face, std::vector<matrix<float, 0, 1>>& face_descriptors, anet_type extrator_descritor_facial);
void extrairFaceThread(const fs::path& fotos, std::vector<matrix<rgb_pixel>>& faces, frontal_face_detector detector_face, shape_predictor detector_pontos);
std::vector<int> fibonnaciSearchAndProcess(std::vector<FuncClust>distancias, int ponteiroInicio, int ponteiroFim, double busca, int ate);
void encontrarPessoa(int posicao, std::vector<FuncClust>funcionarios, bool direcao, std::vector<float>mediasBusca, std::vector<std::string> todosOsCadastros, int idCentroide);
void encontraSimilaridade(FuncClust funcionario, std::vector<float>mediasBusca, std::vector<std::string> todosOsCadastros, int idCentroide);
std::vector<matrix<float, 0, 1>> extrairDescritores(std::vector<matrix<rgb_pixel>> faces, anet_type extrator_descritor_facial);
std::vector<matrix<rgb_pixel>> extrairFace(shape_predictor detector_pontos, std::vector<fs::path> image_paths, frontal_face_detector detector_face);
std::vector<int> possiveisCentroides(std::vector<Centroid>centroides, std::vector<float>dicritoresReduzidos);
std::vector<int> separaValoresRepetidos(std::vector<std::vector<int>>matriz);


int main()
{

    //################################################################
    //########                                                ########
    //########               INICIO DO SISTEMA                ########
    //########                                                ########
    //################################################################            
    int quantFramesKmeansPCA = 5;
    int quantFramesLBPH = 3;
    int qtdParaTreino = qtdMinimaCadastrada;
    int quantImagensMatch = 3;

    //Variáveis Dlib
    frontal_face_detector detector_face = get_frontal_face_detector();
    
    shape_predictor detector_pontos; 
    dlib::deserialize("modelos_treinados\\shape_predictor_68_face_landmarks.dat") >> detector_pontos;

    anet_type extrator_descritor_facial;
    dlib::deserialize("modelos_treinados\\dlib_face_recognition_resnet_model_v1.dat") >> extrator_descritor_facial;
    //------------------------------------------------


    std::string arquivoDeErro = "errosAoCadastrar.txt";

    criarDiretorio(modelosYmlPath);
    criarDiretorio(arquivos);
    std::vector<std::string> todosOsCadastros = carregarTodasPessoas();
    std::vector<string> todasAsPessoasKmeans = carregarPessoasKmeans();
    std::vector<ClustFunc> centClass = carregarCentroidesClassificacoes();

    int escolha = 0;

    while (escolha != 4) {

        bool jaFoiCadastrado = false;

        system("cls");

        printf("Bem vindo ao FaceH");
        printf("Selecione uma opção:\n1 - Treinar modelos\n2 - Realizar testes\n\n");

        std::cin >> escolha;

        if (escolha == 1) {

            //################################################################
            //########                                                ########
            //########              PARTE 1 - CADASTRO                ########
            //########                                                ########
            //################################################################

            system("cls");


            bool continuarCadastro = false;
            bool pegaProximo = false;
            std::vector<Centroid>centroides = carregarCentroides();
            std::vector<Registro>registrosCarregados = carregarDescritores(todosOsCadastros, quantFramesKmeansPCA);

            std::string caminhoRootStr = "D:\\Datasets\\Datasets faciais\\CelebV-HQ\\CelebV_Fotos_Final_V2_Validas";
            fs::path caminhoRoot = caminhoRootStr;

            std::string ultimaPessoa = "";
            if (todosOsCadastros.size() > 0) {
                ultimaPessoa = split(split(todosOsCadastros[todosOsCadastros.size() - 1], "\\")[1], "-")[0];
            }


            if (fs::exists(caminhoRoot) && fs::is_directory(caminhoRoot)) {

                for (const auto& pessoa : fs::directory_iterator(caminhoRoot)) {


                    bool treinamentoVazio = false;
                    ObjTreinamento treinamento = {};

                    std::vector<string> pessoaSplit = split(pessoa.path().string(), "\\");
                    string pessoaNome = pessoaSplit[pessoaSplit.size() - 1];

                    std::string pessoaContent = "Nome: " + pessoaNome + "\n";

                    if (ultimaPessoa != "") {

                        if (pessoaNome == ultimaPessoa) {
                            continuarCadastro = true;
                        }
                    }
                    else {
                        pegaProximo = true;
                    }


                    if (pegaProximo) {

                        //##############################################
                        //######  POPULA O OBJETO DE TREINAMENTO  ######
                        //##############################################

                        //Abrir o arquivo de quantas pessoas estão cadastradas

                        todosOsCadastros = carregarTodasPessoas();
                        todasAsPessoasKmeans = carregarPessoasKmeans();

                        //---------------------------------------------------------

                        //Abrir o arquivo de quantas pessoas novas foram cadastradas                                 
                        std::vector<std::string> recemCadastrados = carregaPessoasNovas();
                        //---------------------------------------------------------



                        treinamento = populaObjTreinamento(pessoa, quantFramesKmeansPCA, todosOsCadastros);

                        if (treinamento.idList.size() > 0) {

                            //Se tiver mais do que tantas pessoas sabemos que 
                            //já existem auto valores e auto vetores cadastrados, 
                            //não precisamos chegar se os arquivos existem ou não.
                            if (todosOsCadastros.size() > qtdMinimaCadastrada) {
                                qtdParaTreino = qtdSuperiorCadastrada;
                            }


                            if (todosOsCadastros.size() >= qtdMinimaCadastrada) {


                                //Verifica se existem uma determinada quantidade de pessoas novas cadastradas
                                //caso exista, deve-se retreinar o k-means
                                if (recemCadastrados.size() >= qtdParaTreino) {


                                    bool detectouFace = salvaDescritores_treinaLBPH(pessoaNome, treinamento, detector_face, detector_pontos, extrator_descritor_facial, quantFramesLBPH);                                    

                                    if (detectouFace) {

                                        recemCadastrados = carregaPessoasNovas();

                                        todosOsCadastros = carregarTodasPessoas();


                                        //Carrega os descritores                                                                                 
                                        std::vector<Registro>descritoresCarregados = carregarDescritores(todosOsCadastros, quantFramesKmeansPCA);
                                        //------------------------------------------------------------

                                        //################################################################
                                        //########                                                ########
                                        //########       REALIZA O TREINAMENTO DO PCA             ########
                                        //########                                                ########
                                        //################################################################


                                        //TREINA O PCA E RETORNA OS VALORES FINAIS                                        
                                        std::vector<std::vector<float>> matrizResultante = treinarPca(descritoresCarregados);
                                        //------------------------------------------------------------


                                        //Salva os descritores reduzidos
                                        auto salvaDescritoresReduzidosInicio = high_resolution_clock::now();

                                        int idRegistroAtual = 0;
                                        int contadorId = 0;
                                        int ultimaPosicao = 0;
                                        int sizeListaCaminho = todosOsCadastros.size();

                                        idRegistroAtual = descritoresCarregados[0].id;


                                        for (int linha = 1; linha < matrizResultante.size(); linha++) {

                                            //Conta quantos descritoresCarregados são de cada pessoa                                            
                                            int idRegistro = descritoresCarregados[linha].id;

                                            if (idRegistro == idRegistroAtual) {
                                                contadorId++;
                                            }
                                            else {
                                                idRegistroAtual = idRegistro;
                                                contadorId = 0;
                                            }
                                            //-----------------------------------------------------------


                                            //Retorna os descritores reduzidos em formato de string
                                            std::string descritorReduzidoContent = descritoresString(matrizResultante[linha]);
                                            //------------------------------------------------------------


                                            ultimaPosicao = salvaDescritoresReduzidos(todosOsCadastros, sizeListaCaminho, idRegistro, contadorId, descritorReduzidoContent, ultimaPosicao);
                                        }


                                        auto salvaDescritoresReduzidosFim = high_resolution_clock::now();
                                        duration<double, std::milli> ms_doubleSalvaDescritoresReduzidos = salvaDescritoresReduzidosFim - salvaDescritoresReduzidosInicio;
                                        pessoaContent += "\nsalvaDescritoresReduzidos: " + std::to_string(ms_doubleSalvaDescritoresReduzidos.count());
                                        //------------------------------------------------------------



                                        //Carrega pessoas que possuem registros reduzidos e normalizados                                        
                                        std::vector<Registro> registrosReduzidos = carregarDescritoresReduzidos(todosOsCadastros, quantFramesKmeansPCA);

                                        std::vector<int>idsQueJaPassaram;
                                        for (int i = 0; i < registrosReduzidos.size(); i++) {

                                            bool jaPassou = false;
                                            for (int j = 0; j < idsQueJaPassaram.size(); j++) {

                                                if (registrosReduzidos[i].id == idsQueJaPassaram[j]) {
                                                    jaPassou = true;
                                                    j = idsQueJaPassaram.size();
                                                }
                                            }

                                            if (!jaPassou) {
                                                std::vector<Registro> registrosClusters;
                                                for (int j = i; j < registrosReduzidos.size(); j++) {

                                                    if (registrosClusters.size() < 1) {
                                                        registrosClusters.push_back(registrosReduzidos[i]);
                                                    }
                                                    else {

                                                        if (registrosReduzidos[j].id == registrosReduzidos[i].id) {
                                                            registrosClusters.push_back(registrosReduzidos[j]);
                                                        }
                                                    }
                                                }
                                                idsQueJaPassaram.push_back(registrosReduzidos[i].id);


                                                //Salva media dos descritores reduzidos
                                                salvaMediasDescritoresReduzidos(registrosClusters);

                                            }
                                        }
                                        //------------------------------------------------------------


                                        //################################################################
                                        //########                                                ########
                                        //########    REALIZA A CLUSTERIZAÇÃO COM O  K-MEANS      ########
                                        //########                                                ########
                                        //################################################################



                                        //Cria centroides
                                        if (!std::filesystem::exists(arquivoCentroides)) {
                                            centroides = criarCentroides(registrosReduzidos);
                                        }
                                        //---------------------------------------------------



                                        //Aplica o algoritmo do K-means e salva os novos centroidss                                        
                                        registrosReduzidos = kmeansFit(centroides, registrosReduzidos);
                                        //---------------------------------------------------


                                        //Remodela os centroides caso a quantidade de pessoas seja
                                        //superior ao minimo de pessoas cadastradas no kmeans                                        
                                        //caso contrário ele só salva as classificações mesmo                                        
                                        if (todasAsPessoasKmeans.size() > qtdMinimaCadastradaRemodelarCentroides) {

                                            todosOsCadastros = carregarTodasPessoas();
                                            CentroidesRegistros centroidesRegistros = remodelarCentroides(todosOsCadastros, quantFramesKmeansPCA);

                                            centroides = centroidesRegistros.centroides;
                                            registrosReduzidos = centroidesRegistros.registros;

                                            fs::remove(pessoasKmeans);
                                        }
                                        else {

                                            //Salva os centroides
                                            salvarClassificacoes(centroides, registrosReduzidos);
                                        }
                                        //---------------------------------------------------                                       

                                        //Apaga os registros dos funcionários novos
                                        std::filesystem::remove(caminhosPessoas);
                                        jaFoiCadastrado = true;
                                    }
                                }
                                else {

                                    //################################################################
                                    //########                                                ########
                                    //########         APLICA O PCA EM NOVO REGISTRO          ########
                                    //########                                                ########
                                    //################################################################                                                                        

                                    bool detectouFace = salvaDescritores_treinaLBPH(pessoaNome, treinamento, detector_face, detector_pontos, extrator_descritor_facial, quantFramesLBPH);


                                    if (detectouFace) {

                                        todosOsCadastros = carregarTodasPessoas();
                                        int idNovaPessoa = treinamento.idList[0];
                                        std::vector<Registro>registrosCarregados = carregarDescritoresById(idNovaPessoa, "descritor", todosOsCadastros, quantFramesKmeansPCA);


                                        registrosCarregados = aplicaPcaEmNovoRegistro(registrosCarregados, pessoaNome);


                                        //################################################################
                                        //########                                                ########
                                        //########       APLICA O KMEANS EM NOVO REGISTRO         ########
                                        //########                                                ########
                                        //################################################################

                                        //SalvarClassificacoes                                        
                                        std::vector<Centroid>idsCentroids;
                                        for (int i = 0; i < registrosCarregados.size(); i++) {
                                            Centroid centroid = classificaRegistro(registrosCarregados[i].dimensoes, centroides);
                                            idsCentroids.push_back(centroid);
                                        }

                                        std::vector<Centroid>centroidesUnicos;

                                        for (int i = 0; i < idsCentroids.size(); i++) {

                                            bool jaExiste = false;
                                            for (int j = 0; j < centroidesUnicos.size(); j++) {

                                                if (centroidesUnicos[j].id == idsCentroids[i].id) {

                                                    if (idsCentroids[i].distancia > centroidesUnicos[j].distancia) {
                                                        idsCentroids[i] = centroidesUnicos[j];
                                                    }
                                                    jaExiste = true;
                                                }
                                            }
                                            if (!jaExiste) {

                                                Centroid centroidUnico = {};
                                                centroidUnico.id = idsCentroids[i].id;
                                                centroidUnico.distancia = idsCentroids[i].distancia;
                                                centroidUnico.dimensoes = idsCentroids[i].dimensoes;

                                                centroidesUnicos.push_back(centroidUnico);
                                            }
                                        }


                                        std::vector<Registro>registros;
                                        registros.reserve(centroidesUnicos.size());

                                        for (int i = 0; i < centroidesUnicos.size(); i++) {
                                            Registro reg;
                                            reg.id = registrosCarregados[0].id;
                                            reg.classe = centroidesUnicos[i];

                                            registros.push_back(reg);
                                        }
                                        
                                        salvarClassificacoesPorRegistro(registros, todosOsCadastros);
                                        //---------------------------------------------------

                                        recemCadastrados = carregaPessoasNovas();

                                        jaFoiCadastrado = true;
                                    }
                                }
                            }


                            if (recemCadastrados.size() < qtdMinimaCadastrada && !jaFoiCadastrado) {

                                salvaDescritores_treinaLBPH(pessoaNome, treinamento, detector_face, detector_pontos, extrator_descritor_facial, quantFramesLBPH);
                                
                            }
                            else {

                                jaFoiCadastrado = false;
                            }
                        }


                    }

                    if (continuarCadastro) {
                        pegaProximo = true;
                    }
                }
            }
            else {
                std::cout << "Caminho inválido ou não é uma pasta." << std::endl;
            }
        }
        if (escolha == 2) {


            //################################################################
            //########                                                ########
            //########      PARTE 2 - Teste de reconhecimento         ########
            //########                                                ########
            //################################################################

            system("cls");

            std::string pessoaContent = "";
            std::vector<Centroid>centroides = carregarCentroides();
            todosOsCadastros = carregarTodasPessoas();


            std::string caminhoTesteStr = "D:\\Datasets\\Datasets faciais\\CelebV-HQ\\CelebV_Fotos_Final_V2_Validas";
            fs::path caminhoTeste = caminhoTesteStr;
            std::string indexAnterior = "";
            std::vector<std::vector<std::string>> listaGeralDeImagens;


            if (fs::exists(caminhoTeste) && fs::is_directory(caminhoTeste)) {




                for (const auto& pessoaTeste : fs::directory_iterator(caminhoTeste)) {

                    bool continuaComparando = true;
                    std::string ultimaFotoPath = "";

                    while (continuaComparando) {

                        if (fs::exists(pessoaTeste) && fs::is_directory(caminhoTeste)) {

                            std::cout << pessoaTeste << "\n";
                            pessoaContent = "pessoa: " + pessoaTeste.path().string() + "\n";

                            auto geralInicio = high_resolution_clock::now();

                            //Extrai faces
                            auto extraiFaceInicio = high_resolution_clock::now();

                            std::vector<Centroid>idsCentroids;
                            std::vector<fs::path> image_paths;
                            bool continuaRecolhendoFotos = false;

                            int qtdDeImagens = 0;
                            for (const auto& fotos : fs::directory_iterator(pessoaTeste)) {

                                if (ultimaFotoPath != "") {

                                    if (continuaRecolhendoFotos) {

                                        if (qtdDeImagens < quantImagensMatch) {
                                            image_paths.push_back(fotos.path());
                                            qtdDeImagens += 1;
                                        }
                                        else {
                                            break;
                                        }
                                    }

                                    if (fotos.path().string() == ultimaFotoPath) {
                                        continuaRecolhendoFotos = true;
                                    }
                                }
                                else {

                                    if (qtdDeImagens < quantImagensMatch) {
                                        image_paths.push_back(fotos.path());
                                        qtdDeImagens += 1;
                                    }
                                    else {
                                        break;
                                    }
                                }
                            }


                            if (image_paths.size() < quantImagensMatch) {
                                continuaComparando = false;                                
                            }
                            else {
                                ultimaFotoPath = image_paths[image_paths.size() - 1].string();
                            }



                            std::vector<matrix<rgb_pixel>> faces = extrairFace(detector_pontos, image_paths, detector_face);


                            auto extraiFaceFim = high_resolution_clock::now();
                            duration<double, std::milli> ms_doubleextraiFace = extraiFaceFim - extraiFaceInicio;
                            pessoaContent += "\nextraiFace: " + std::to_string(ms_doubleextraiFace.count());
                            //-------------------------------------------------------



                            //Extrai os descritores dos rostos e para cada rosto um arquivo txt onde contém os descritores é criado.
                            auto extraiDescritoresInicio = high_resolution_clock::now();

                            std::vector<matrix<float, 0, 1>> face_descriptors = extrairDescritores(faces, extrator_descritor_facial);

                            auto extraiDescritoresFim = high_resolution_clock::now();
                            duration<double, std::milli> ms_doubleExtraiDescritores = extraiDescritoresFim - extraiDescritoresInicio;
                            pessoaContent += "\nextraiDescritores: " + std::to_string(ms_doubleExtraiDescritores.count());



                            std::vector<std::vector<float>> descritoresAll;
                            std::vector<std::vector<int>>centroidesMutuos;
                            for (int indexDesc = 0; indexDesc < face_descriptors.size(); indexDesc++) {


                                Registro registro;
                                registro.id = 0;

                                for (long j = 0; j < face_descriptors[indexDesc].nr(); ++j) {
                                    registro.dimensoes.push_back(face_descriptors[indexDesc](j));
                                }


                                auto reduzDimensaoInicio = high_resolution_clock::now();

                                std::vector<float> dicritoresReduzidos = reduzDimensao(registro);
                                descritoresAll.push_back(dicritoresReduzidos);

                                auto reduzDimensaoFim = high_resolution_clock::now();
                                duration<double, std::milli> ms_doublReduzDimensao = reduzDimensaoFim - reduzDimensaoInicio;
                                pessoaContent += "\nreduzDimensao: " + std::to_string(ms_doublReduzDimensao.count());



                                auto classificaRegistroInicio = high_resolution_clock::now();

                                Centroid centroid = classificaRegistro(dicritoresReduzidos, centroides);
                                idsCentroids.push_back(centroid);


                                auto classificaRegistroFim = high_resolution_clock::now();
                                duration<double, std::milli> ms_doublClassificaRegistro = classificaRegistroFim - classificaRegistroInicio;
                                pessoaContent += "\nclassificaRegistro: " + std::to_string(ms_doublClassificaRegistro.count());
                            }


                            //Separa os ids dos centroides com base na similaridade dos cossenos
                            std::vector<int>idCentroidesUnicos = separaValoresRepetidos(centroidesMutuos);
                            //-------------------------------------------------------


                            std::vector<float>mediasDescritoresPessoaAtual;

                            if (descritoresAll.size() > 0) {

                                for (int i = 0; i < descritoresAll[0].size(); i++) {

                                    float media = 0;
                                    for (int j = 0; j < descritoresAll.size(); j++) {

                                        media += descritoresAll[j][i];
                                    }
                                    mediasDescritoresPessoaAtual.push_back(media / descritoresAll[0].size());
                                }
                            }
                            //-------------------------------------------------------


                            //Verifica e extrai quais os ids dos clusters que aparece para cada imagem clusterizada
                            auto centroidesUnicosInicio = high_resolution_clock::now();

                            std::vector<Centroid>centroidesUnicos;                            


                            for (int i = 0; i < idsCentroids.size(); i++) {

                                bool jaExiste = false;
                                for (int j = 0; j < centroidesUnicos.size(); j++) {

                                    if (centroidesUnicos[j].id == idsCentroids[i].id) {

                                        if (idsCentroids[i].distancia < centroidesUnicos[j].distancia) {
                                            centroidesUnicos[j] = idsCentroids[i];
                                        }
                                        jaExiste = true;
                                    }
                                }
                                if (!jaExiste) {

                                    Centroid centroidUnico = {};
                                    centroidUnico.id = idsCentroids[i].id;
                                    centroidUnico.distancia = idsCentroids[i].distancia;
                                    centroidUnico.dimensoes = idsCentroids[i].dimensoes;

                                    centroidesUnicos.push_back(centroidUnico);
                                }
                            }


                            std::vector<Centroid>centroidesUnicosView = centroidesUnicos;

                            auto centroidesUnicosFim = high_resolution_clock::now();
                            duration<double, std::milli> ms_doubleCentroidesUnicos = centroidesUnicosFim - centroidesUnicosInicio;
                            pessoaContent += "\ncentroidesUnicos: " + std::to_string(ms_doubleCentroidesUnicos.count());
                            //-------------------------------------------------------


                            //Match
                            encontrouFuncionario = false;

                            auto matchInicio = high_resolution_clock::now();
                            for (int i = 0; i < centClass.size(); i++) {

                                for (int j = 0; j < centroidesUnicos.size(); j++) {

                                    if (centroidesUnicos[j].id == centClass[i].idClust) {


                                        ClustFunc centClassView = centClass[i];
                                        Centroid centroideUnicoView = centroidesUnicos[j];
                                        std::vector<Centroid> centroidesUnicosView = centroidesUnicos;
                                        std::vector<Centroid>idsCentroidsView = idsCentroids;
                                        //------------------------------------------------------------------


                                        std::vector<int> posicoes = fibonnaciSearchAndProcess(centClass[i].funcs, 0, 0, centroidesUnicos[j].distancia, centClass[i].funcs.size());

                                        if (posicoes.size() == 2) {


                                            std::thread t1(encontrarPessoa, posicoes[0], centClass[i].funcs, false, mediasDescritoresPessoaAtual, todosOsCadastros, centroidesUnicos[j].id);
                                            std::thread t2(encontrarPessoa, posicoes[1], centClass[i].funcs, true, mediasDescritoresPessoaAtual, todosOsCadastros, centroidesUnicos[j].id);

                                            t1.join();
                                            t2.join();

                                        }
                                        else if (posicoes.size() == 1) {

                                            if (posicoes[0] == 0) {
                                                encontrarPessoa(posicoes[0], centClass[i].funcs, true, mediasDescritoresPessoaAtual, todosOsCadastros, centroidesUnicos[j].id);
                                            }
                                            else if (posicoes[0] == centClass[i].funcs.size() - 1) {
                                                encontrarPessoa(posicoes[0], centClass[i].funcs, false, mediasDescritoresPessoaAtual, todosOsCadastros, centroidesUnicos[j].id);
                                            }
                                            else {
                                                encontraSimilaridade(centClass[i].funcs[posicoes[0]], mediasDescritoresPessoaAtual, todosOsCadastros, centroidesUnicos[j].id);

                                                if (!encontrouFuncionario) {

                                                    int posicaoRoot = posicoes[0];
                                                    posicoes.clear();

                                                    posicoes.push_back(posicaoRoot - 1);
                                                    posicoes.push_back(posicaoRoot + 1);

                                                    std::thread t1(encontrarPessoa, posicoes[0], centClass[i].funcs, false, mediasDescritoresPessoaAtual, todosOsCadastros, centroidesUnicos[j].id);
                                                    std::thread t2(encontrarPessoa, posicoes[1], centClass[i].funcs, true, mediasDescritoresPessoaAtual, todosOsCadastros, centroidesUnicos[j].id);

                                                    t1.join();
                                                    t2.join();
                                                }
                                            }
                                        }

                                        if (encontrouFuncionario) {

                                            FuncionarioEncontrado func;
                                            func = funcFind;

                                            //std::cout << "Centroid: " << func.idCentroide << "\n";
                                            //std::cout << "Distancia do centroide: " << func.distancia << "\n\n";
                                            std::cout << "Similaridade: " << func.similaridade << "\n";
                                            std::cout << "Id da pessoa encontrada: " << func.idFuncionario << "\n\n";

                                            pessoaContent += "\nsimilaridade: " + to_string(func.similaridade);
                                            pessoaContent += "\nid: " + to_string(func.idFuncionario);


                                            j = centroidesUnicos.size();

                                            continuaComparando = false;
                                        }
                                    }
                                }
                                if (encontrouFuncionario) {
                                    i = centClass.size();
                                }
                            }


                            auto matchFim = high_resolution_clock::now();
                            duration<double, std::milli> ms_match = matchFim - matchInicio;
                            pessoaContent += "\nmatch: " + std::to_string(ms_match.count());
                            //-------------------------------------------------------


                            auto geralFim = high_resolution_clock::now();
                            duration<double, std::milli> ms_geral = geralFim - geralInicio;
                            pessoaContent += "\ngeral: " + std::to_string(ms_geral.count());
                            pessoaContent += "\n";


                            if (!continuaComparando) {

                                escreverArquivo("Arquivos\\logsMatch.txt", pessoaContent, false);
                            }
                        }
                    }                    
                }
            }            
        }
    }
}

std::string fibonnaciSearch(std::vector<std::string> todosOsRegistros, int splitInicio, int splitFim, int idPesquisado) {

    std::string retorno = "";
    int lenTotal = 0;

    if (splitFim == 0) {
        lenTotal = todosOsRegistros.size();
    }
    else {
        lenTotal = splitFim - splitInicio;
    }

    if (lenTotal % 2 == 0) {
        splitFim = (lenTotal / 2) + splitInicio;
    }
    else {
        splitFim = (floor(lenTotal / 2)) + splitInicio;
    }

    int id = stoi(split(todosOsRegistros[splitFim], "-")[1]);

    if (idPesquisado == id) {
        retorno = todosOsRegistros[splitFim];
    }
    else if (idPesquisado > id) {

        if (splitFim + 1 < todosOsRegistros.size()) {
            int idSeguinte = stoi(split(todosOsRegistros[splitFim + 1], "-")[1]);

            if (idPesquisado == idSeguinte) {
                retorno = todosOsRegistros[splitFim + 1];
            }
            else {

                splitInicio = splitFim;
                splitFim = todosOsRegistros.size();
                retorno = fibonnaciSearch(todosOsRegistros, splitInicio, splitFim, idPesquisado);
            }
        }
        else {
            retorno = "";
        }

    }
    else if (idPesquisado < id) {
        if (splitFim != 0) {
            retorno = fibonnaciSearch(todosOsRegistros, splitInicio, splitFim, idPesquisado);
        }
        else {
            retorno = "";
        }

    }

    return retorno;
}


void salvaMediasDescritoresReduzidos(std::vector<Registro> registrosClusters) {

    std::vector<float> mediasReduzidas;
    for (int i = 0; i < registrosClusters[0].dimensoes.size(); i++) {

        float mediaReduzida = 0;
        for (int j = 0; j < registrosClusters.size(); j++) {

            mediaReduzida += registrosClusters[j].dimensoes[i];
        }
        mediasReduzidas.push_back(mediaReduzida / registrosClusters.size());
    }

    std::string mediasReduzidasContent = "";

    for (int i = 0; i < mediasReduzidas.size(); i++) {

        mediasReduzidasContent += to_string(mediasReduzidas[i]) + "{";
    }

    std::string mediasReduzidasPath = registrosClusters[0].caminhoDoArquivo + "\\descMediasReduzidas.txt";
    escreverArquivo(mediasReduzidasPath, mediasReduzidasContent, true);
}

ObjTreinamento populaObjTreinamento(auto& pessoa, int quantFramesKmeansPCA, std::vector<std::string> todosOsCadastros) {

    //Essa struct é exclusiva para o treinamento do LBPH
    ObjTreinamento treinamento = {};

    //Seleciona as fotos de treinamento randomicamente
    std::vector<string> listaDeFotos;
    std::vector<string> fotosParaTreinamento;

    for (const auto& foto : fs::directory_iterator(pessoa)) {
        listaDeFotos.push_back(foto.path().string());
    }

    if (listaDeFotos.size() > quantFramesKmeansPCA) {

        int min = 0, max = listaDeFotos.size() - 1;

        for (int i = 0; i < quantFramesKmeansPCA; i++) {

            int fotoRand = (std::rand() % (max - min + 1)) + min;
            fotosParaTreinamento.push_back(listaDeFotos[fotoRand]);
        }
    }
    else {
        fotosParaTreinamento = listaDeFotos;
    }


    //Após os frames serem separados, agora popula o objeto que será utilizado
    //para realizar o treinamento com o nome e os descritores de cada frame.
    for (int indTrainImg = 0; indTrainImg < fotosParaTreinamento.size(); indTrainImg++) {

        cv::Mat imagem = cv::imread(fotosParaTreinamento[indTrainImg]);

        std::vector<Deteccao> deteccoes = deteccaoSSD(network, imagem, 150, 0.7f);

        for (int i = 0; i < deteccoes.size(); i++) {

            if (validaDeteccao(deteccoes[i], confiancaMinimaDeteccao, larguraIdeal, alturaIdeal)) {

                treinamento.idList.push_back(todosOsCadastros.size() + 1);
                treinamento.faceList.push_back(deteccoes[i].deteccoes[0]);
                treinamento.caminhoImagens.push_back(fotosParaTreinamento[indTrainImg]);
            }
        }
    }

    return treinamento;
}



//##############################################
//###  CRIA A PASTA PARA PESSOA CADASTRADA  ####
//###      SALVA TODOS OS DESCRITORES       ####
//###             TREINA O LBPH             ####
//##############################################
bool salvaDescritores_treinaLBPH(std::string nomePasta, ObjTreinamento treinamento, frontal_face_detector detector_face, shape_predictor detector_pontos, anet_type extrator_descritor_facial, int quantFramesLBPH) {


    //##############################################
    //###  CRIA A PASTA PARA PESSOA CADASTRADA  ####
    //##############################################
    std::string modeloDiretorio = modelosYmlPath + "\\" + nomePasta;
    std::string dadosPessoais = modeloDiretorio + "\\" + "dados.txt";
    criarDiretorio(modeloDiretorio);


    //##############################################
    //########     SALVA OS DESCRITORES     ########
    //##############################################
    bool detectouFace = salvarDescritores(treinamento, detector_face, detector_pontos, extrator_descritor_facial, modeloDiretorio);


    if (detectouFace) {

        //##############################################
        //########   REALIZA TREINAMENTO LBPH   ########
        //##############################################        
        realizarTreinamentoLBPH(treinamento, quantFramesLBPH, modeloDiretorio);

    }
    else {
        auto ret = rmdir(modeloDiretorio.c_str());
    }    

    return detectouFace;
}


bool salvarDescritores(ObjTreinamento treinamento, frontal_face_detector detector_face, shape_predictor detector_pontos, anet_type extrator_descritor_facial, string modeloDiretorio) {

    bool detectouFace = false;

    //Intera com a face da pessoa
    std::vector<matrix<rgb_pixel>> faces;
    for (int i = 0; i < treinamento.faceList.size(); i++) {

        cv::Mat imagem = cv::imread(treinamento.caminhoImagens[i]);

        //Converte o Image do OpenCv para o Dlib
        dlib::array2d<bgr_pixel> dlibImg;
        dlib::assign_image(dlibImg, dlib::cv_image<bgr_pixel>(imagem));

        //Os rostos são convertidos em um formato que dê para ser lido pelo descritor
        int qtdFaces = 0;
        for (auto face : detector_face(dlibImg)) {

            auto pontos = detector_pontos(dlibImg, face);
            matrix<rgb_pixel> face_chip;
            dlib::extract_image_chip(dlibImg, get_face_chip_details(pontos, 150, 0.25), face_chip);
            faces.push_back(std::move(face_chip));
            qtdFaces += 1;
        }

        if (qtdFaces > 1) {
            faces.clear();
        }

        if (faces.size() > 0) {
            detectouFace = true;
        }

    }

    //Extrai os descritores dos rostos e para cada rosto um arquivo txt onde contém os descritores é criado.
    if (faces.size() > 0) {

        std::vector<matrix<float, 0, 1>> face_descriptors = extrator_descritor_facial(faces);
        for (int indexDesc = 0; indexDesc < face_descriptors.size(); indexDesc++) {

            std::string descritor = modeloDiretorio + "\\" + "descritor" + std::to_string(indexDesc) + ".txt";
            std::string descritorContent = "";

            for (long j = 0; j < face_descriptors[indexDesc].nr(); ++j) {
                descritorContent += to_string(face_descriptors[indexDesc](j)) + "{";
            }
            //Grava os descritores                
            escreverArquivo(descritor, descritorContent, false);
        }


        std::string contentMedia = "";
        if (face_descriptors.size() > 0) {

            for (size_t i = 0; i < face_descriptors[0].size(); ++i) {

                float media = 0;
                for (size_t j = 0; j < face_descriptors.size(); ++j) {

                    media += face_descriptors[j](i);
                }
                media = media / face_descriptors[0].size();

                contentMedia += to_string(media) + "{";
            }
        }

        std::string descritorMedias = modeloDiretorio + "\\descMedias.txt";

        escreverArquivo(descritorMedias, contentMedia, false);
    }


    return detectouFace;
}


void realizarTreinamentoLBPH(ObjTreinamento treinamento, int quantFramesLBPH, string modeloDiretorio) {

    std::string dadosPessoais = modeloDiretorio + "\\" + "dados.txt";
    std::vector<cv::Mat> faceList;
    std::vector<int>idList;

    if (treinamento.faceList.size() > quantFramesLBPH) {

        int minFace = 0, maxFace = treinamento.faceList.size() - 1;

        for (int indQuant = 0; indQuant < quantFramesLBPH; indQuant++) {

            int faceRand = (std::rand() % (maxFace - minFace + 1)) + minFace;
            faceList.push_back(treinamento.faceList[faceRand]);
            idList.push_back(treinamento.idList[faceRand]);
        }
    }
    else {
        faceList = treinamento.faceList;
        idList = treinamento.idList;
    }

    cv::Ptr <cv::face::FaceRecognizer> lbphClassifier = cv::face::LBPHFaceRecognizer::create();


    //Define o caminho do modelo LBPH da pessoa para o a pasta que foi criada
    std::string modeloYml = modeloDiretorio + "\\" + to_string(treinamento.idList[0]) + "_LBPH.yml";

    //Treina o modelo LBPH
    lbphClassifier->train(faceList, idList);
    lbphClassifier->write(modeloYml);


    //Grava o modelo no arquivo de modelos novos
    escreverArquivo(caminhosPessoas, modeloDiretorio, false);

    //Grava o modelo no arquivo de todos os modelos
    escreverArquivo(todasAsPessoas, modeloDiretorio + "-" + to_string(treinamento.idList[0]), false);

    //Grava o controle de realocação dos centroides do k-means
    escreverArquivo(pessoasKmeans, modeloDiretorio, false);


    escreverArquivo(dadosPessoais, "IdPessoa[" + to_string(treinamento.idList[0]), false);
}


void salvarClassificacoes(std::vector<Centroid>centroides, std::vector<Registro>todosOsRegistros) {

    std::string contentString;

    std::vector<ClustFunc>clustFuncList;

    for (int i = 0; i < centroides.size(); i++) {
        ClustFunc clust;
        clust.idClust = centroides[i].id;
        clustFuncList.push_back(clust);
    }

    for (int j = 0; j < todosOsRegistros.size(); j++) {

        std::vector<Centroid> clusters;

        for (int k = j; k < todosOsRegistros.size(); k++) {

            if (todosOsRegistros[j].id == todosOsRegistros[k].id) {

                if (clusters.size() > 0) {

                    bool clusterSeparado = false;
                    for (int l = 0; l < clusters.size(); l++) {


                        if (todosOsRegistros[k].classe.id == clusters[l].id) {

                            if (todosOsRegistros[k].classe.distancia < clusters[l].distancia) {
                                clusters[l] = todosOsRegistros[k].classe;
                            }

                            clusterSeparado = true;
                        }
                    }
                    if (!clusterSeparado) {
                        clusters.push_back(todosOsRegistros[k].classe);
                    }
                }
                else {
                    clusters.push_back(todosOsRegistros[k].classe);
                }
            }
            else {

                for (int h = 0; h < clusters.size(); h++) {

                    for (int i = 0; i < clustFuncList.size(); i++) {

                        if (clustFuncList[i].idClust == clusters[h].id) {

                            FuncClust func;
                            func.idFunc = todosOsRegistros[j].id;
                            func.distancia = clusters[h].distancia;
                            clustFuncList[i].funcs.push_back(func);
                        }
                    }
                }

                for (int i = 0; i < clustFuncList.size(); i++) {
                    clustFuncList[i].funcs = quickSort(clustFuncList[i].funcs, 0, clustFuncList[i].funcs.size() - 1);
                }
                

                j = k - 1;
                k = todosOsRegistros.size();
                clusters.clear();
            }
        }
    }


    for (int i = 0; i < clustFuncList.size(); i++) {
        std::string content = "Centroide[" + to_string(clustFuncList[i].idClust);

        for (int j = 0; j < clustFuncList[i].funcs.size(); j++) {
            
            content += "{"+to_string(clustFuncList[i].funcs[j].idFunc) + "|" + to_string(clustFuncList[i].funcs[j].distancia);            
        }

        contentString += content + "\n";
    }    

    escreverArquivo(centroidesClassificacoes, contentString, true);
}


void salvarClassificacoesPorRegistro(std::vector<Registro>registros, std::vector<string> todosOsRegistros) {

    std::string content = "";

    std::vector<Centroid>centroides;

    if (std::filesystem::exists(centroidesClassificacoes)) {

        std::fstream arquivo;
        arquivo.open(centroidesClassificacoes, std::ios::in);

        std::vector<std::string> listaDeCaminhos;
        if (arquivo.is_open()) {

            std::string sa;

            while (getline(arquivo, sa)) {


                bool concatenouContent = false;
                std::vector<string>partes = splitText('[', sa);
                std::vector<string>valores = split(partes[1], "{");

                int idCluster = stoi(valores[0]);

                for (int i = 0; i < registros.size(); i++) {

                    if (registros[i].classe.id == idCluster) {

                        std::vector<FuncClust>funcionarios;
                        for (int j = 1; j < valores.size(); j++) {

                            std::vector<string>atributos = split(valores[j], "|");

                            FuncClust funcClust;
                            funcClust.idFunc = stoi(atributos[0]);
                            funcClust.distancia = stof(atributos[1]);

                            funcionarios.push_back(funcClust);
                        }


                        for (int j = 0; j < funcionarios.size(); j++) {

                            
                            Registro descMediasReduzidasCarregado = carregarMedias(funcionarios[j].idFunc, "descMediasReduzidas", todosOsRegistros);

                            if (descMediasReduzidasCarregado.dimensoes.size() > 0) {
                                
                                Centroid centroidComNovaDistancia = calculoDistancia(descMediasReduzidasCarregado.dimensoes, registros[i].classe);
                                funcionarios[j].distancia = centroidComNovaDistancia.distancia;
                            }                            
                        }

                        funcionarios = quickSort(funcionarios, 0, funcionarios.size()-1);
                       

                        bool encontrouPosicao = false;
                        sa = "Centroide["+ valores[0];
                        for (int j = 0; j < funcionarios.size(); j++) {

                            if (!encontrouPosicao) {
                                if (registros[i].classe.distancia < funcionarios[j].distancia) {
                                    encontrouPosicao = true;
                                    sa += "{" + to_string(registros[i].id) + "|" + to_string(registros[i].classe.distancia);                                    
                                }                                
                                sa += "{" + to_string(funcionarios[j].idFunc) + "|" + to_string(funcionarios[j].distancia);
                                
                            }
                            else {
                                sa += "{" + to_string(funcionarios[j].idFunc) + "|" + to_string(funcionarios[j].distancia);
                            }
                            
                        }

                        if (!encontrouPosicao) {
                            sa += "{" + to_string(registros[i].id) + "|" + to_string(registros[i].classe.distancia);
                        }

                        concatenouContent = true;

                        content += sa + "\n";
                    }
                }

                if (!concatenouContent) {
                    content += sa + "\n";
                }
            }
            arquivo.close();
        }
    }

    escreverArquivo(centroidesClassificacoes, content, true);
}


std::vector<Registro> aplicaPcaEmNovoRegistro(std::vector<Registro>registrosCarregados, std::string pessoaNome) {

    //Carrega autovetores
    std::vector<std::vector<float>> autoVetores = carregaAutoVetores();

    //Carrega autovalores                                            
    std::vector<float> autoValores = carregaArquivo(eigenVectorsFile);

    //Carrega as médias dos descritores
    std::vector<float>mediaAutoVetores = carregaArquivo(mediasFile);

    //Carrega os desvios padrões
    std::vector<float>desviosPadroes = carregaArquivo(desviosPadroesFile);


    std::vector< std::vector<float>> dimensoesReduzidas;
    dimensoesReduzidas.reserve(registrosCarregados.size());


    for (int l = 0; l < registrosCarregados.size(); l++) {

        std::vector<float> dimensaoSubtraida;

        for (int indexDimen = 0; indexDimen < registrosCarregados[l].dimensoes.size(); indexDimen++) {

            dimensaoSubtraida.push_back((registrosCarregados[l].dimensoes[indexDimen] - mediaAutoVetores[indexDimen]) / desviosPadroes[indexDimen]);
        }

        std::vector<float> projecoes;
        for (int indexVet = 0; indexVet < autoVetores.size(); indexVet++) {

            float projecao = 0;
            for (int indexValor = 0; indexValor < autoVetores[indexVet].size(); indexValor++) {
                projecao += dimensaoSubtraida[indexValor] * autoVetores[indexVet][indexValor];
            }
            projecoes.push_back(projecao);
        }


        std::vector<float> projecoesReduzidas;
        projecoesReduzidas.reserve(numComponents);

        for (int indexProj = 0; indexProj < projecoes.size(); indexProj++) {

            if (indexProj < numComponents) {
                projecoesReduzidas.push_back(projecoes[indexProj]);
            }
            else {
                indexProj = projecoes.size();
            }
        }

        std::string projecaoContent = "";
        for (int indexDimen = 0; indexDimen < projecoesReduzidas.size(); indexDimen++) {
            projecaoContent += to_string(projecoesReduzidas[indexDimen]) + "{";
        }

        std::string descritorFile = "\\descReduzido" + to_string(l) + ".txt";
        std::string caminho = modelosYmlPath + "\\" + pessoaNome;

        escreverArquivo(caminho + descritorFile, projecaoContent, false);


        registrosCarregados[l].dimensoes.clear();
        registrosCarregados[l].dimensoes.reserve(projecoesReduzidas.size());

        registrosCarregados[l].dimensoes = projecoesReduzidas;
    }

    return registrosCarregados;
}


std::vector<float> reduzDimensao(Registro registro) {


    //Carrega autovetores
    std::vector<std::vector<float>> autoVetores = carregaAutoVetores();

    std::vector<std::vector<float>> autoVetoresTranspostos;
    for (int coluna = 0; coluna < autoVetores.size(); coluna++) {

        std::vector<float> valores;
        for (int linha = 0; linha < autoVetores[coluna].size(); linha++) {
            valores.push_back(autoVetores[linha][coluna]);
        }
        autoVetoresTranspostos.push_back(valores);
    }




    //Carrega as médias dos descritores
    std::vector<float>mediaAutoVetores = carregaArquivo(mediasFile);

    //Carrega desvios padrões
    std::vector<float>desviosPadroes = carregaArquivo(desviosPadroesFile);


    std::vector<float> dimensaoSubtraida;

    for (int indexDimen = 0; indexDimen < registro.dimensoes.size(); indexDimen++) {

        dimensaoSubtraida.push_back((registro.dimensoes[indexDimen] - mediaAutoVetores[indexDimen]) / desviosPadroes[indexDimen]);
    }


    std::vector<float> projecoes;
    for (int indexLinha = 0; indexLinha < autoVetoresTranspostos.size(); indexLinha++) {

        float projecao = 0;
        for (int indexColuna = 0; indexColuna < autoVetoresTranspostos[indexLinha].size(); indexColuna++) {
            projecao += dimensaoSubtraida[indexColuna] * autoVetoresTranspostos[indexLinha][indexColuna];
        }
        projecoes.push_back(projecao);
    }


    std::vector<float> projecoesReduzidas;
    for (int indexProj = 0; indexProj < projecoes.size(); indexProj++) {

        if (indexProj < numComponents) {
            projecoesReduzidas.push_back(projecoes[indexProj]);
        }
        else {
            indexProj = projecoes.size();
        }
    }

    return projecoesReduzidas;
}


void salvaClasseDeCadaCadastro(std::vector<Registro> registrosReduzidos) {

    std::vector<int> jaPassou;
    for (int indexReg = 0; indexReg < registrosReduzidos.size(); indexReg++) {

        //Verifica se o registro já passou
        bool passou = false;
        for (int indexPassou = 0; indexPassou < jaPassou.size(); indexPassou++) {

            if (registrosReduzidos[indexReg].id == jaPassou[indexPassou]) {
                passou = true;
            }
        }
        //--------------------------------------

        if (!passou) {


            //Separa os registros que são da mesma pessoa (vão existir registros de várias pessoas)                                                    
            std::vector<Registro> registrosIguais;
            registrosIguais.push_back(registrosReduzidos[indexReg]);

            if (indexReg + 1 < registrosReduzidos.size()) {

                int count = 0;
                for (int indexSub = indexReg + 1; indexSub < registrosReduzidos.size(); indexSub++) {

                    if (registrosReduzidos[indexReg].id == registrosReduzidos[indexSub].id) {
                        registrosIguais.push_back(registrosReduzidos[indexSub]);
                        count++;
                    }

                    if (count == 2) {
                        indexSub = registrosReduzidos.size();
                    }
                }
            }
            jaPassou.push_back(registrosReduzidos[indexReg].id);


            //Analisa a quantidade que cada classe foi escolhida para determinada pessoa
            std::vector<int>idClasse;
            std::vector<int>qtd;
            std::vector<int>passouId;

            for (int indexIguais = 0; indexIguais < registrosIguais.size(); indexIguais++) {

                bool existe = false;

                for (int indexPassou = 0; indexPassou < passouId.size(); indexPassou++) {

                    if (registrosIguais[indexIguais].classe.id == passouId[indexPassou]) {
                        existe = true;
                        indexPassou = passouId.size();
                    }
                }

                if (!existe) {

                    int contador = 1;
                    if (indexIguais + 1 < registrosIguais.size()) {

                        for (int indexIguais2 = indexIguais + 1; indexIguais2 < registrosIguais.size(); indexIguais2++) {

                            if (registrosIguais[indexIguais].classe.id == registrosIguais[indexIguais2].classe.id) {
                                contador++;
                            }
                        }
                    }

                    idClasse.push_back(registrosIguais[indexIguais].classe.id);
                    qtd.push_back(contador);
                    passouId.push_back(registrosIguais[indexIguais].classe.id);
                }
            }


            //Escolhe a classe que teve maior frequência de escolha
            int maior = 0;
            int idMaior = 0;

            for (int indexQtd = 0; indexQtd < qtd.size(); indexQtd++) {

                if (indexQtd == 0) {
                    maior = qtd[indexQtd];
                    idMaior = indexQtd;
                }
                else {
                    if (qtd[indexQtd] > maior) {
                        maior = qtd[indexQtd];
                        idMaior = indexQtd;
                    }
                }
            }

            std::string dadosPessoa = registrosReduzidos[indexReg].caminhoDoArquivo + "\\dados.txt";
            int maiorClasse = idClasse[idMaior];

            substituirValorArquivo(dadosPessoa, "Classe[" + to_string(maiorClasse));
        }
    }
}


std::vector<Registro> kmeansFit(std::vector<Centroid>centroides, std::vector<Registro> registrosReduzidos) {

    bool alterouAlgumCentroid = true;

    while (alterouAlgumCentroid) {

        registrosReduzidos = classificaRegistros(registrosReduzidos, centroides);

        std::vector<Centroid> novosCentroides;
        std::vector<Registro>jaPassou;

        for (int indexReg = 0; indexReg < registrosReduzidos.size(); indexReg++) {

            bool passou = false;
            for (int indexPassou = 0; indexPassou < jaPassou.size(); indexPassou++) {

                if (jaPassou[indexPassou].classe.id == registrosReduzidos[indexReg].classe.id) {
                    passou = true;
                }
            }

            if (!passou) {


                std::vector<Registro> registrosCluster = separaOsClusters(registrosReduzidos, indexReg);

                if (registrosCluster.size() > 0) {


                    RetornoCentroides retorno = reposicionaClusters(registrosCluster, centroides);
                    alterouAlgumCentroid = retorno.alterou;

                    centroides = retorno.centroides;
                }
            }

            jaPassou.push_back(registrosReduzidos[indexReg]);
        }
    }




    salvaCentroides(centroides);

    return registrosReduzidos;
}

int kmeansPredict(std::vector<Centroid>centroides, Registro registro) {

    std::vector<Registro>registros;
    registros.push_back(registro);

    registros = classificaRegistros(registros, centroides);

    return registros[0].classe.id;
}


std::vector<std::string> carregaPessoasNovas() {

    std::fstream file;
    file.open(caminhosPessoas, std::ios::in);
    std::vector<std::string> listaDeCaminhos;

    if (file.is_open()) {
        string tp;
        while (getline(file, tp)) {
            listaDeCaminhos.push_back(tp);
        }
        file.close();
    }

    return listaDeCaminhos;
}

std::vector<std::vector<float>> carregaAutoVetores() {

    std::fstream eigenVectorsStream;
    eigenVectorsStream.open(eigenVectorsFile, std::ios::in);
    std::vector<std::vector<float>> autoVetores;

    if (eigenVectorsStream.is_open()) {
        string tp;
        while (getline(eigenVectorsStream, tp)) {

            std::vector<std::string> valores = splitText('{', tp);
            std::vector<float>valoresF;

            for (int i = 0; i < valores.size(); i++) {
                valoresF.push_back(stof(valores[i]));
            }
            autoVetores.push_back(valoresF);
        }
        eigenVectorsStream.close();
    }

    return autoVetores;
}


std::vector<float> carregaArquivo(std::string arquivo) {

    std::fstream eigenValuestream;
    eigenValuestream.open(arquivo, std::ios::in);
    std::vector<float> autoValores;

    if (eigenValuestream.is_open()) {
        string tp;
        while (getline(eigenValuestream, tp)) {

            autoValores.push_back(stof(tp));
        }
        eigenValuestream.close();
    }

    return autoValores;
}




std::vector<std::string> carregarTodasPessoas() {

    std::fstream todasPessoasFile;
    todasPessoasFile.open(todasAsPessoas, std::ios::in);
    std::vector<std::string> todasPessoas;

    if (todasPessoasFile.is_open()) {
        string tp;
        while (getline(todasPessoasFile, tp)) {
            if (tp != "") {
                todasPessoas.push_back(tp);
            }
        }
        todasPessoasFile.close();
    }

    return todasPessoas;
}


std::vector<std::string> carregarPessoasKmeans() {

    std::vector<std::string> todasPessoas;

    if (std::filesystem::exists(pessoasKmeans)) {

        std::fstream pessoasKmeansFile;
        pessoasKmeansFile.open(pessoasKmeans, std::ios::in);

        if (pessoasKmeansFile.is_open()) {
            string tp;
            while (getline(pessoasKmeansFile, tp)) {
                if (tp != "") {
                    todasPessoas.push_back(tp);
                }
            }
            pessoasKmeansFile.close();
        }
    }

    return todasPessoas;
}



std::vector<Registro> carregarDescritores(std::vector<string>todosOsCadastros, int quantFramesKmeansPCA) {

    std::vector<Registro> registros;
    std::vector<std::vector<float>> allDescritores;

    for (int i = 0; i < todosOsCadastros.size(); i++) {

        std::vector<string> cadastrosSplit = split(todosOsCadastros[i], "-");

        for (int j = 0; j < quantFramesKmeansPCA; j++) {


            std::string caminho = cadastrosSplit[0] + "\\descritor" + to_string(j) + ".txt";
            std::vector<float>descritorVec;

            if (std::filesystem::exists(caminho)) {


                std::ifstream arquivo(caminho, std::ios::in | std::ios::ate);
                if (!arquivo.is_open()) {
                    std::cerr << "Erro ao abrir o arquivo!" << std::endl;
                }

                std::streampos fileSize = arquivo.tellg();
                arquivo.seekg(0, std::ios::beg);

                std::string fileContent(fileSize, '\0');
                fileContent.resize(fileSize);
                arquivo.read(&fileContent[0], fileSize);
                arquivo.close();

                std::vector<std::string> tokens = split(fileContent, "{");
                descritorVec.reserve(tokens.size());

                for (const auto& token : tokens) {
                    if (!token.empty()) {
                        descritorVec.push_back(std::stof(token));
                    }
                }

                Registro registroPessoa = {};
                registroPessoa.nome = cadastrosSplit[0];
                registroPessoa.caminhoDoArquivo = cadastrosSplit[0];
                registroPessoa.id = stoi(cadastrosSplit[1]);
                registroPessoa.dimensoes = descritorVec;
                registros.push_back(registroPessoa);
            }
        }
    }

    return registros;
}


std::vector<Registro> carregarDescritoresById(int idNovaPessoa, std::string tipoDescritor, std::vector<std::string> todosOsRegistros, int quantFramesKmeansPCA) {

    std::vector<Registro> registros;

    std::string caminhoPessoa = "";


    caminhoPessoa = fibonnaciSearch(todosOsRegistros, 0, 0, idNovaPessoa);


    std::vector<std::string>caminhoSplit = split(caminhoPessoa, "-");


    for (int j = 1; j < quantFramesKmeansPCA; j++) {


        std::string caminho = caminhoSplit[0] + "\\" + tipoDescritor + to_string(j) + ".txt";
        std::vector<float>descritorVec;

        if (std::filesystem::exists(caminho)) {

            Registro registroPessoa = {};
            registroPessoa.id = stoi(caminhoSplit[1]);

            std::vector<float>descritorVec;

            std::ifstream arquivo(caminho, std::ios::in | std::ios::ate);
            if (!arquivo.is_open()) {
                std::cerr << "Erro ao abrir o arquivo!" << std::endl;
            }

            std::streampos fileSize = arquivo.tellg();
            arquivo.seekg(0, std::ios::beg);

            std::string fileContent(fileSize, '\0');
            fileContent.resize(fileSize);
            arquivo.read(&fileContent[0], fileSize);
            arquivo.close();

            std::vector<std::string> tokens = split(fileContent, "{");
            descritorVec.reserve(tokens.size());

            for (const auto& token : tokens) {
                if (!token.empty()) {
                    descritorVec.push_back(std::stof(token));
                }
            }

            registroPessoa.dimensoes = descritorVec;
            registros.push_back(registroPessoa);
        }
    }
    return registros;
}

Registro carregarMedias(int idNovaPessoa, std::string tipoDescritor, std::vector<std::string> todosOsRegistros) {

    Registro registro;
    std::string caminhoPessoa = "";

    caminhoPessoa = fibonnaciSearch(todosOsRegistros, 0, 0, idNovaPessoa);


    if (caminhoPessoa != "") {
        std::vector<std::string>caminhoSplit = split(caminhoPessoa, "-");

        std::string caminho = caminhoSplit[0] + "\\" + tipoDescritor + ".txt";
        std::vector<float>descritorVec;

        if (std::filesystem::exists(caminho)) {


            registro.id = stoi(caminhoSplit[1]);

            std::vector<float>descritorVec;

            std::ifstream arquivo(caminho, std::ios::in | std::ios::ate);
            if (!arquivo.is_open()) {
                std::cerr << "Erro ao abrir o arquivo!" << std::endl;
            }

            std::streampos fileSize = arquivo.tellg();
            arquivo.seekg(0, std::ios::beg);

            std::string fileContent(fileSize, '\0');
            fileContent.resize(fileSize);
            arquivo.read(&fileContent[0], fileSize);
            arquivo.close();

            std::vector<std::string> tokens = split(fileContent, "{");
            descritorVec.reserve(tokens.size());

            for (const auto& token : tokens) {
                if (!token.empty()) {
                    descritorVec.push_back(std::stof(token));
                }
            }

            registro.dimensoes = descritorVec;

        }
    }
    
    return registro;
}



std::vector<Registro> carregarDescritoresReduzidos(std::vector<string>todosOsCadastros, int quantFramesKmeansPCA) {

    std::vector<Registro> registros;
    std::vector<std::vector<float>> allDescritores;

    for (int i = 0; i < todosOsCadastros.size(); i++) {

        std::vector<string> cadastrosSplit = split(todosOsCadastros[i], "-");

        for (int j = 1; j < quantFramesKmeansPCA; j++) {


            std::string caminho = cadastrosSplit[0] + "\\descReduzido" + to_string(j) + ".txt";
            std::vector<float>descritorVec;

            if (std::filesystem::exists(caminho)) {

                std::vector<float>descritorVec;

                std::ifstream arquivo(caminho, std::ios::in | std::ios::ate);
                if (!arquivo.is_open()) {
                    std::cerr << "Erro ao abrir o arquivo!" << std::endl;
                }

                std::streampos fileSize = arquivo.tellg();
                arquivo.seekg(0, std::ios::beg);

                std::string fileContent(fileSize, '\0');
                fileContent.resize(fileSize);
                arquivo.read(&fileContent[0], fileSize);
                arquivo.close();

                std::vector<std::string> tokens = split(fileContent, "{");
                descritorVec.reserve(tokens.size());

                for (const auto& token : tokens) {
                    if (!token.empty()) {
                        descritorVec.push_back(std::stof(token));
                    }
                }

                Registro registroPessoa = {};
                registroPessoa.nome = cadastrosSplit[0];
                registroPessoa.caminhoDoArquivo = cadastrosSplit[0];
                registroPessoa.id = stoi(cadastrosSplit[1]);
                registroPessoa.dimensoes = descritorVec;
                registros.push_back(registroPessoa);
            }
        }
    }

    return registros;
}

int getIdFromTxt(std::string pessoaPath) {

    int registroPessoaId = 0;

    for (const auto& arquivoTxt : fs::directory_iterator(pessoaPath)) {


        if (containString("dados", arquivoTxt.path().string().c_str())) {

            std::fstream arquivo;
            arquivo.open(arquivoTxt, std::ios::in);

            if (arquivo.is_open()) {

                std::string linha;

                while (getline(arquivo, linha)) {

                    if (linha != "" && containString("IdPessoa", linha)) {
                        std::vector<string> linhaSplit = split(linha, "[");

                        if (linhaSplit.size() > 0) {
                            registroPessoaId = stoi(linhaSplit[1]);
                            break;
                        }
                    }
                }
                arquivo.close();
            }
        }
    }

    return registroPessoaId;
}


std::vector<std::vector<float>> carregaDescritoresFromTxt(std::string pessoaPath, std::string tipoDescritor) {

    std::vector<std::vector<float>>allDescritores;

    for (const auto& arquivoTxt : fs::directory_iterator(pessoaPath)) {

        std::vector<float>descritorVec;
        if (containString(".txt", arquivoTxt.path().string().c_str()) && containString(tipoDescritor, arquivoTxt.path().string().c_str())) {

            std::fstream arquivo;
            arquivo.open(arquivoTxt, std::ios::in);

            if (arquivo.is_open()) {

                std::string linha;

                while (getline(arquivo, linha)) {

                    if (linha != "") {
                        descritorVec.push_back(stof(trim(linha)));
                    }
                }
                arquivo.close();
            }
        }
        if (descritorVec.size() > 0) {
            allDescritores.push_back(descritorVec);
        }
    }


    return allDescritores;
}


std::vector<Centroid> carregarCentroides() {

    std::vector<Centroid>centroides;

    if (std::filesystem::exists(arquivoCentroides)) {

        std::fstream arquivo;
        arquivo.open(arquivoCentroides, std::ios::in);

        std::vector<std::string> listaDeCaminhos;
        if (arquivo.is_open()) {

            std::string sa;

            while (getline(arquivo, sa)) {

                std::vector<string>partes = splitText('[', sa);
                std::vector<string>valores = splitText('{', partes[2]);

                std::vector<float>valoresFloat;

                for (int i = 0; i < valores.size(); i++) {
                    valoresFloat.push_back(stof(valores[i]));
                }

                Centroid centroide = { stoi(partes[1]),valoresFloat,-1 };
                centroides.push_back(centroide);
            }
            arquivo.close();
        }
    }
    return centroides;
}


void atualizarCentroides(std::vector<Centroid> centroidesAtualizados) {

    std::string novoArquivo;
    int idCentroidNovo = 0;

    if (std::filesystem::exists(arquivoCentroides)) {

        std::string sa;

        std::fstream arquivo;
        arquivo.open(arquivoCentroides, std::ios::in);

        std::vector<std::string> listaDeCaminhos;
        if (arquivo.is_open()) {

            while (getline(arquivo, sa)) {

                if (sa != "") {

                    std::vector<string>partes = splitText('[', sa);
                    std::vector<string>valores = splitText('{', partes[2]);

                    bool encontrou = false;

                    for (int i = 0; i < centroidesAtualizados.size(); i++) {

                        if (stoi(partes[1]) == centroidesAtualizados[i].id) {

                            encontrou = true;

                            std::string novaLinha = "Centroide[" + to_string(centroidesAtualizados[i].id) + "[";
                            for (int j = 0; j < centroidesAtualizados[i].dimensoes.size(); j++) {

                                if (j < centroidesAtualizados[i].dimensoes.size() - 1) {

                                    novaLinha += to_string(centroidesAtualizados[i].dimensoes[j]) + "{";
                                }
                                else {
                                    novaLinha += to_string(centroidesAtualizados[i].dimensoes[j]);
                                }
                            }
                            if (arquivo.peek() != EOF) {
                                novoArquivo += novaLinha + "\n";
                            }
                            else {
                                novoArquivo += novaLinha;
                            }
                        }
                    }

                    if (!encontrou) {

                        if (arquivo.peek() != EOF) {
                            novoArquivo += sa + "\n";
                        }
                        else {
                            novoArquivo += sa;
                        }
                    }
                }
            }
            arquivo.close();
        }
        escreverArquivo(arquivoCentroides, novoArquivo, true);

    }
}

void adicionarCentroides(Centroid centroideAtualizado) {

    std::string conteudo = "Centroide[" + to_string(centroideAtualizado.id) + "[";

    for (int j = 0; j < centroideAtualizado.dimensoes.size(); j++) {

        if (j < centroideAtualizado.dimensoes.size() - 1) {
            conteudo += to_string(centroideAtualizado.dimensoes[j]) + "{";
        }
        else {
            conteudo += to_string(centroideAtualizado.dimensoes[j]);
        }
    }

    escreverArquivo(arquivoCentroides, conteudo, false);
}


void atualizarCentroidesClassificacoes(std::vector<ClustFunc>classificacoes, std::string path) {

    std::string novoArquivo;

    if (std::filesystem::exists(centroidesClassificacoes)) {

        std::string sa;


        std::fstream arquivo;
        arquivo.open(centroidesClassificacoes, std::ios::in);

        std::vector<std::string> listaDeCaminhos;
        if (arquivo.is_open()) {

            while (getline(arquivo, sa)) {

                if (sa != "") {

                    std::vector<string>partes = splitText('[', sa);
                    std::vector<string>valores = splitText('{', partes[1]);
                    bool encontrou = false;

                    for (int i = 0; i < classificacoes.size(); i++) {

                        if (stoi(valores[0]) == classificacoes[i].idClust) {

                            encontrou = true;

                            std::string novaLinha = "Centroide[" + to_string(classificacoes[i].idClust) + "{";
                            for (int j = 0; j < classificacoes[i].funcs.size(); j++) {

                                if (j < classificacoes[i].funcs.size() - 1) {

                                    novaLinha += to_string(classificacoes[i].funcs[j].distancia) + "|" + to_string(classificacoes[i].funcs[j].distancia) + "{";
                                }
                                else {
                                    novaLinha += to_string(classificacoes[i].funcs[j].distancia) + "|" + to_string(classificacoes[i].funcs[j].distancia);
                                }
                            }
                            novoArquivo += novaLinha + "\n";

                        }
                    }

                    if (!encontrou) {
                        novoArquivo += sa + "\n";
                    }
                }
            }
            arquivo.close();
        }
        escreverArquivo(path, novoArquivo, true);

    }
}



std::vector<ClustFunc> carregarCentroidesClassificacoes() {

    std::vector<ClustFunc> clustersCentroides;

    if (std::filesystem::exists(centroidesClassificacoes)) {

        std::fstream arquivo;
        arquivo.open(centroidesClassificacoes, std::ios::in);

        std::vector<std::string> listaDeCaminhos;
        if (arquivo.is_open()) {

            std::string sa;

            while (getline(arquivo, sa)) {

                if (sa != "") {

                    std::vector<string>partes = splitText('[', sa);
                    std::vector<string>valores = splitText('{', partes[1]);

                    struct ClustFunc clustFunc = {};
                    clustFunc.idClust = stof(valores[0]);

                    std::vector<FuncClust>funcionarios;
                    for (int i = 1; i < valores.size(); i++) {

                        std::vector<string>id_distancia = split(valores[i], "|");

                        FuncClust funcClust = {};
                        funcClust.idFunc = stoi(id_distancia[0]);
                        funcClust.distancia = stof(id_distancia[1]);

                        funcionarios.push_back(funcClust);
                    }

                    clustFunc.funcs = funcionarios;
                    clustersCentroides.push_back(clustFunc);
                }
            }
            arquivo.close();
        }
    }
    return clustersCentroides;
}


bool containString(std::string pesquisa, std::string texto) {

    bool retorno = false;
    bool achouPrimeiroCaractere = false;
    int pos = 0;
    int acertos = 0;


    for (int i = 0; i < texto.size(); i++) {

        if (acertos == pesquisa.size()) {
            retorno = true;
            i = texto.size();
        }


        if (achouPrimeiroCaractere) {
            if (pesquisa[pos] == texto[i]) {
                acertos++;
                pos++;
            }
        }
        else {
            pos = 0;
            acertos = 0;
            achouPrimeiroCaractere = false;
        }


        if (!achouPrimeiroCaractere) {
            if (pesquisa[pos] == texto[i]) {
                achouPrimeiroCaractere = true;
                pos++;
                acertos++;
            }
        }
    }

    if (acertos == pesquisa.size()) {
        retorno = true;
    }


    return retorno;
}


std::vector<std::vector<float>> treinarPca(std::vector<Registro> registrosCarregados) {

    std::vector<std::vector<float>> retorno;

    std::vector<std::vector<float>> matrizDeDescritores;

    for (int i = 0; i < registrosCarregados.size(); i++) {

        std::vector<float> vetorDescritores;
        for (int j = 0; j < registrosCarregados[i].dimensoes.size(); j++) {

            vetorDescritores.push_back(registrosCarregados[i].dimensoes[j]);
        }
        matrizDeDescritores.push_back(vetorDescritores);
    }

    retorno = pcaFit(matrizDeDescritores, false);

    return retorno;
}


std::string descritoresString(std::vector<float> matrizResultanteLinha) {

    std::string descritorReduzidoContent = "";

    for (int coluna = 0; coluna < numComponents; coluna++) {
        descritorReduzidoContent += to_string(matrizResultanteLinha[coluna]) + "{";
    }

    return descritorReduzidoContent;
}


int salvaDescritoresReduzidos(std::vector<std::string> listaDeCaminhos, int sizeListaCaminhos, int idRegistro, int contadorId, std::string descritorReduzidoContent, int ultimaPosicao) {

    std::string caminhoSelecionado = "";

    for (int indexCaminho = ultimaPosicao; indexCaminho < sizeListaCaminhos; indexCaminho++) {

        std::vector<std::string> splitLista = split(listaDeCaminhos[indexCaminho], "-");

        int idSearch = stoi(splitLista[1]);

        if (idRegistro == idSearch) {
            caminhoSelecionado = splitLista[0];
            ultimaPosicao = indexCaminho;
            indexCaminho = sizeListaCaminhos;
        }
    }

    std::string descReduzidoFile = caminhoSelecionado +
        "\\descReduzido" +
        std::to_string(contadorId) +
        ".txt";

    escreverArquivo(descReduzidoFile, descritorReduzidoContent, true);

    return ultimaPosicao;
}


std::string getIdPessoa(std::string dadosPessoais) {

    std::string idPessoa = "";

    std::fstream dadosFile;
    dadosFile.open(dadosPessoais, std::ios::in);
    if (dadosFile.is_open()) {

        std::string linha;

        while (getline(dadosFile, linha)) {

            linha = trim(linha);
            if (linha != "") {

                std::vector<string>partes = splitText('[', linha);

                if (partes[0] == "IdPessoa") {
                    idPessoa = partes[1];
                }
            }
        }
        dadosFile.close();
    }

    return idPessoa;
}



void substituirValorArquivo(std::string nomeArquivo, auto conteudo) {

    std::string newFile = "";
    bool temClasse = false;

    if (std::filesystem::exists(nomeArquivo)) {

        std::fstream file;
        file.open(nomeArquivo, std::ios::in);


        if (file.is_open()) {
            string tp;
            while (getline(file, tp)) {

                if (!containString("Classe", tp)) {
                    newFile += tp + "\n";
                }
                else {
                    newFile += conteudo + "\n";
                    temClasse = true;
                }
            }
            file.close();
        }
    }

    if (temClasse == false) {
        newFile += conteudo + "\n";
    }


    std::ofstream arquivoCadastro(nomeArquivo);
    arquivoCadastro << newFile;
    arquivoCadastro.close();
}


std::vector<FuncClust> swap(std::vector<FuncClust>arr, int pos1, int pos2) {

    FuncClust funcClust = arr[pos1];

    arr[pos1] = arr[pos2];
    arr[pos2] = funcClust;

    return arr;
}


std::vector<double> retornaMedia(std::vector<std::vector<double>>vetores) {

    std::vector<double> medias;


    for (int linha = 0; linha < vetores[0].size(); linha++) {

        double somatoria = 0;
        for (int coluna = 0; coluna < vetores.size(); coluna++) {

            somatoria += vetores[coluna][linha];
        }

        medias.push_back(somatoria / vetores.size());
    }
    return medias;
}


int confereTamanhoMatriz(std::vector<std::vector<float>> vetores) {

    int tamanhoVetores = 0;
    for (int indVet = 0; indVet < vetores.size(); indVet++) {

        if (indVet == 0) {
            tamanhoVetores = vetores[indVet].size();
        }
        else {
            if (vetores[indVet].size() != tamanhoVetores) {
                tamanhoVetores = 0;
                indVet = vetores.size();
            }
        }
    }
    return tamanhoVetores;
}

float similaridaDeCosseno(std::vector<std::vector<float>> vetores, int tamanhoVetores) {

    std::vector<float>somas;
    float divisor = 0;

    for (int indexVec = 0; indexVec < vetores.size(); indexVec++) {

        float soma = 0;
        for (int i = 0; i < vetores[indexVec].size(); i++) {

            soma += pow(vetores[indexVec][i], 2);
        }
        somas.push_back(sqrt(soma));
    }

    float multiplicacoes = somas[0];
    for (int i = 1; i < somas.size(); i++) {
        multiplicacoes *= somas[i];
    }

    float somatoria = 0;
    for (int indexVet = 0; indexVet < tamanhoVetores; indexVet++) {

        float multiplicacao = 0;
        for (int i = 0; i < vetores.size(); i++) {

            if (i == 0) {
                multiplicacao = vetores[i][indexVet];
            }
            else {
                multiplicacao *= vetores[i][indexVet];
            }

        }
        somatoria += multiplicacao;
    }
    return somatoria / multiplicacoes;
}


float normDifference(const std::vector<float>vec1, const std::vector<float>vec2) {
    // Verifica se os dois vetores têm o mesmo tamanho
    if (vec1.size() != vec2.size()) {
        std::cerr << "Erro: Os vetores têm tamanhos diferentes." << std::endl;
        return -1; // ou algum outro valor para indicar erro
    }

    float sumOfSquares = 0.0;

    // Calcula a soma dos quadrados das diferenças dos elementos correspondentes dos dois vetores
    for (size_t i = 0; i < vec1.size(); ++i) {
        float diff = vec1[i] - vec2[i];
        sumOfSquares += diff * diff;
    }

    // Retorna a raiz quadrada da soma dos quadrados
    return std::sqrt(sumOfSquares);
}




std::vector<FuncClust> swapVector(std::vector<FuncClust>vetor, int pos1, int pos2) {

    FuncClust posAux = vetor[pos1];

    vetor[pos1] = vetor[pos2];
    vetor[pos2] = posAux;

    return vetor;
}


RetPartition partition(std::vector<FuncClust>arr, int low, int high)
{
    FuncClust pivot = arr[high];

    int i = (low - 1);

    for (int j = low; j <= high; j++)
    {

        if (arr[j].distancia < pivot.distancia)
        {

            i++;
            arr = swapVector(arr, i, j);
        }
    }
    arr = swapVector(arr, i + 1, high);

    RetPartition retPartition = { i + 1,arr };

    return retPartition;
}


std::vector<FuncClust> quickSort(std::vector<FuncClust>arr, int low, int high)
{

    if (low < high)
    {
        RetPartition retPartition = partition(arr, low, high);

        int pi = retPartition.partition;

        arr = retPartition.vetor;

        arr = quickSort(arr, low, pi - 1);
        arr = quickSort(arr, pi + 1, high);
    }

    return arr;
}


std::vector<Centroid>criarCentroides(std::vector<Registro> registrosReduzidos) {


    std::vector<Centroid> centroides;
    std::vector<float>menores;
    std::vector<float>maiores;

    for (int coluna = 0; coluna < registrosReduzidos[0].dimensoes.size(); coluna++) {

        std::vector<float>colunaAtual;
        for (int linha = 0; linha < registrosReduzidos.size(); linha++) {

            colunaAtual.push_back(registrosReduzidos[linha].dimensoes[coluna]);
        }

        float maior = 0;
        float menor = 0;

        for (int i = 0; i < colunaAtual.size(); i++) {
            if (i == 0) {
                maior = colunaAtual[i];
                menor = colunaAtual[i];
            }
            else {
                if (colunaAtual[i] > maior) {
                    maior = colunaAtual[i];
                }
                if (colunaAtual[i] < menor) {
                    menor = colunaAtual[i];
                }
            }
        }
        menores.push_back(menor);
        maiores.push_back(maior);
    }




    std::srand(time(0));

    for (int indCent = 0; indCent < qtdCentroides; indCent++) {

        std::vector<float>dimensoes;

        for (int i = 0; i < menores.size(); i++) {

            float r = (float)std::rand() / (float)RAND_MAX;
            float random = menores[i] + r * (maiores[i] - menores[i]);

            dimensoes.push_back(random);

        }
        Centroid centroid = { indCent,dimensoes,-1 };
        centroides.push_back(centroid);
    }


    for (int indCent = 0; indCent < centroides.size(); indCent++) {

        std::string conteudo = "Centroide" + std::string("[") + std::to_string(indCent) + "{";

        for (int indexDimen = 0; indexDimen < centroides[indCent].dimensoes.size(); indexDimen++) {

            if (indexDimen < centroides[indCent].dimensoes.size() - 1) {

                conteudo += std::to_string(centroides[indCent].dimensoes[indexDimen]) + "{";
            }
            else {

                conteudo += std::to_string(centroides[indCent].dimensoes[indexDimen]);
            }
        }

        escreverArquivoKmeans(arquivoCentroides, conteudo);
    }

    return centroides;
}





std::vector<ClustFunc> carregarCentroidesClassificacoesTeste(std::string path) {

    std::vector<ClustFunc> clustersCentroides;

    if (std::filesystem::exists(path)) {

        std::fstream arquivo;
        arquivo.open(path, std::ios::in);

        std::vector<std::string> listaDeCaminhos;
        if (arquivo.is_open()) {

            std::string sa;

            while (getline(arquivo, sa)) {

                if (sa != "") {

                    std::vector<string>partes = splitText('[', sa);
                    std::vector<string>valores = splitText('{', partes[1]);

                    struct ClustFunc clustFunc = {};
                    clustFunc.idClust = stof(valores[0]);

                    std::vector<FuncClust>funcionarios;
                    for (int i = 1; i < valores.size(); i++) {

                        std::vector<string>id_distancia = split(valores[i], "|");

                        FuncClust funcClust = {};
                        funcClust.idFunc = stoi(id_distancia[0]);
                        funcClust.distancia = stof(id_distancia[1]);

                        funcionarios.push_back(funcClust);
                    }

                    clustFunc.funcs = funcionarios;
                    clustersCentroides.push_back(clustFunc);
                }
            }
            arquivo.close();
        }
    }
    return clustersCentroides;
}


std::vector<Centroid> carregarCentroidesByIds(std::vector<int> idsCentroides, std::string path) {

    std::vector<Centroid>centroides;

    if (std::filesystem::exists(path)) {

        std::fstream arquivo;
        arquivo.open(path, std::ios::in);

        std::vector<std::string> listaDeCaminhos;
        if (arquivo.is_open()) {

            std::string sa;
            for (int i = 0; i < idsCentroides.size(); i++) {

                while (getline(arquivo, sa)) {

                    std::vector<string>partes = splitText('[', sa);

                    if (idsCentroides[i] == std::stoi(partes[1])) {
                        std::vector<string>valores = splitText('{', partes[2]);

                        std::vector<float>valoresFloat;

                        for (int i = 0; i < valores.size(); i++) {
                            valoresFloat.push_back(stof(valores[i]));
                        }

                        Centroid centroide = { stoi(partes[1]),valoresFloat,-1 };
                        centroides.push_back(centroide);

                        break;
                    }
                }
            }
            arquivo.close();
        }
    }
    return centroides;
}


Centroid carregarCentroideById(int idCentroide) {

    Centroid centroide;

    if (std::filesystem::exists(arquivoCentroides)) {

        std::fstream arquivo;
        arquivo.open(arquivoCentroides, std::ios::in);

        std::vector<std::string> listaDeCaminhos;
        if (arquivo.is_open()) {

            std::string sa;


            while (getline(arquivo, sa)) {

                std::vector<string>partes = splitText('[', sa);

                if (idCentroide == std::stoi(partes[1])) {
                    std::vector<string>valores = splitText('{', partes[2]);

                    std::vector<float>valoresFloat;

                    for (int i = 0; i < valores.size(); i++) {
                        valoresFloat.push_back(stof(valores[i]));
                    }

                    centroide.id = stoi(partes[1]);
                    centroide.dimensoes = valoresFloat;
                    centroide.distancia = -1;

                    break;
                }
            }

            arquivo.close();
        }
    }
    return centroide;
}


std::vector<Registro> kmeansFitTeste(std::vector<Centroid>centroides, std::vector<Registro> registrosReduzidos) {

    bool alterouAlgumCentroid = true;

    while (alterouAlgumCentroid) {

        registrosReduzidos = classificaRegistros(registrosReduzidos, centroides);

        std::vector<Centroid> novosCentroides;
        std::vector<Registro>jaPassou;

        for (int indexReg = 0; indexReg < registrosReduzidos.size(); indexReg++) {

            bool passou = false;
            for (int indexPassou = 0; indexPassou < jaPassou.size(); indexPassou++) {

                if (jaPassou[indexPassou].classe.id == registrosReduzidos[indexReg].classe.id) {
                    passou = true;
                }
            }

            if (!passou) {


                std::vector<Registro> registrosCluster = separaOsClusters(registrosReduzidos, indexReg);

                if (registrosCluster.size() > 0) {


                    RetornoCentroides retorno = reposicionaClusters(registrosCluster, centroides);
                    alterouAlgumCentroid = retorno.alterou;

                    centroides = retorno.centroides;
                }
            }

            jaPassou.push_back(registrosReduzidos[indexReg]);
        }
    }


    //Contar quantos registros tem por cluster
    /*
    std::vector<ClustQuant>clustQuantList;
    std::vector<int>idClusterList;
    for (int i = 0; i < registrosReduzidos.size(); i++) {
        idClusterList.push_back(registrosReduzidos[i].classe.id);
    }

    std::vector<int>jaPassou;
    for (int i = 0; i < idClusterList.size(); i++) {

        bool passou = false;
        for (int j = 0; j < jaPassou.size(); j++) {

            if (idClusterList[i] == jaPassou[j]) {
                passou = true;
            }

        }

        if (!passou) {
            int quantidade = 0;
            for (int j = i + 1; j < idClusterList.size(); j++) {

                if (idClusterList[i] == idClusterList[j]) {
                    quantidade++;
                }
            }

            jaPassou.push_back(idClusterList[i]);
            ClustQuant clustQuant = { idClusterList[i] ,quantidade + 1 };
            clustQuantList.push_back(clustQuant);
        }
    }
    */

    return registrosReduzidos;
}


Centroid redefinePosicaoCentroide(Centroid centroideCheio, Centroid centroideReposicionado) {

    for (int k = 0; k < centroideCheio.dimensoes.size(); k++) {

        double percent = centroideCheio.dimensoes[k] * 0.4;

        if (centroideReposicionado.dimensoes[k] > 0) {
            centroideReposicionado.dimensoes[k] = centroideCheio.dimensoes[k] + percent;
        }
        else {
            percent = std::sqrt(std::pow(percent * 0.2, 2));
            centroideReposicionado.dimensoes[k] = centroideCheio.dimensoes[k] - percent;
        }
    }

    return centroideReposicionado;
}


void salvaCentroides(std::vector<Centroid> centroides) {

    for (int indexCent = 0; indexCent < centroides.size(); indexCent++) {

        std::string conteudo = "Centroide" + std::string("[") + std::to_string(indexCent) + "[";


        for (int indexDimen = 0; indexDimen < centroides[indexCent].dimensoes.size(); indexDimen++) {

            if (indexDimen < centroides[indexCent].dimensoes.size() - 1) {

                conteudo += std::to_string(centroides[indexCent].dimensoes[indexDimen]) + "{";
            }
            else {

                conteudo += std::to_string(centroides[indexCent].dimensoes[indexDimen]);
            }
        }

        escreverArquivo(arquivoCentroidesAux, conteudo, false);
    }

    std::filesystem::remove(arquivoCentroides);
    std::filesystem::rename(arquivoCentroidesAux, arquivoCentroides);
}


CentroidesRegistros remodelarCentroides(std::vector<string> todosOsRegistros, int quantFramesKmeansPCA) {


    std::vector<ClustFunc> clustersTeste = carregarCentroidesClassificacoes();

    std::vector<ClustFunc> clustersAcima;
    std::vector<ClustFunc> clustersZerados;
    std::vector<int>clustersJaUtilizados;

    for (int i = 0; i < clustersTeste.size(); i++) {
        if (clustersTeste[i].funcs.size() > qtdMinimaPorCluster) {
            clustersAcima.push_back(clustersTeste[i]);
        }

        if (clustersTeste[i].funcs.size() == 0) {
            clustersZerados.push_back(clustersTeste[i]);
        }
    }


    //Cria clusters caso não tenha clusters vazios o suficiente
    std::vector<int>clustersCriados;
    if (clustersAcima.size() > clustersZerados.size()) {

        int qtdClustersParaCriar = clustersAcima.size() - clustersZerados.size();
        for (int i = 0; i < qtdClustersParaCriar; i++) {
            if (clustersCriados.size() == 0) {

                clustersCriados.push_back(clustersTeste[clustersTeste.size() - 1].idClust + 1);
            }
            else {

                clustersCriados.push_back(clustersCriados[clustersCriados.size() - 1] + 1);
            }
        }
    }

    std::vector<int>idsClustersVaziosQueJaPassaram;
    std::vector<int>idsClustersCriadosQueJaPassaram;

    for (int i = 0; i < clustersAcima.size(); i++) {


        Centroid centroideCheio = carregarCentroideById(clustersAcima[i].idClust);
        std::vector<Registro>registrosCarregados;

        std::vector<int>idsDeFuncionarios;

        //Carrega todos os registros de todos os funcionarios do centroid
        for (int j = 0; j < clustersAcima[i].funcs.size(); j++) {
            idsDeFuncionarios.push_back(clustersAcima[i].funcs[j].idFunc);
            std::vector<Registro> registrosPesquisados = carregarDescritoresById(clustersAcima[i].funcs[j].idFunc, "descReduzido", todosOsRegistros, quantFramesKmeansPCA);

            for (int k = 0; k < registrosPesquisados.size(); k++) {
                registrosCarregados.push_back(registrosPesquisados[k]);
            }
        }


        //Caso o oposto a isso ocorra, sabemos que todos os clusters zerados 
        //já foram utilizados, então temos que utilizar os clusters que foram criados.
        if (clustersZerados.size() > 0 && idsClustersVaziosQueJaPassaram.size() < clustersZerados.size()) {


            //Altera a posição do cluster zerado do cluster lotado alterando 20% o valor de cada dimensão.
            for (int j = 0; j < clustersZerados.size(); j++) {

                //Verifica se o cluster zerado já foi utilizado
                bool jaUtilizado = false;
                for (int k = 0; k < idsClustersVaziosQueJaPassaram.size(); k++) {

                    if (idsClustersVaziosQueJaPassaram[k] == clustersZerados[j].idClust) {
                        jaUtilizado = true;
                        k = idsClustersVaziosQueJaPassaram.size();
                    }
                }

                if (!jaUtilizado) {

                    Centroid centroideZerado = carregarCentroideById(clustersZerados[j].idClust);


                    centroideZerado = redefinePosicaoCentroide(centroideCheio, centroideZerado);


                    std::vector<Centroid>centroides;
                    centroides.push_back(centroideCheio);
                    centroides.push_back(centroideZerado);


                    std::vector<Centroid> centroidesCarregados = carregarCentroides();

                    for (int k = 0; k < centroidesCarregados.size(); k++) {

                        for (int l = 0; l < centroides.size(); l++) {

                            if (centroidesCarregados[k].id == centroides[l].id) {
                                centroidesCarregados[k] = centroides[l];
                            }
                        }
                    }


                    registrosCarregados = kmeansFitTeste(centroides, registrosCarregados);
                    atualizarCentroides(centroides);


                    idsClustersVaziosQueJaPassaram.push_back(clustersZerados[j].idClust);
                }
            }
        }

        else {

            for (int j = 0; j < clustersCriados.size(); j++) {

                bool jaUtilizado = false;
                for (int k = 0; k < idsClustersCriadosQueJaPassaram.size(); k++) {

                    if (clustersCriados[j] == idsClustersCriadosQueJaPassaram[k]) {
                        jaUtilizado = true;
                        k = idsClustersCriadosQueJaPassaram.size();
                    }
                }

                if (!jaUtilizado) {

                    Centroid centroideCriado = centroideCheio;

                    centroideCriado.id = clustersCriados[j];
                    centroideCriado.dimensoes = centroideCheio.dimensoes;
                    centroideCriado.distancia = centroideCheio.distancia;


                    centroideCriado = redefinePosicaoCentroide(centroideCheio, centroideCriado);

                    std::vector<Centroid>centroides;
                    centroides.push_back(centroideCheio);
                    centroides.push_back(centroideCriado);

                    registrosCarregados = kmeansFitTeste(centroides, registrosCarregados);
                    adicionarCentroides(centroideCriado);

                    idsClustersCriadosQueJaPassaram.push_back(clustersCriados[j]);
                    j = clustersCriados.size();
                }
            }
        }
    }


    std::vector<Centroid>centroides = carregarCentroides();
    std::vector<Registro>descritoresReduzidosCarregados = carregarDescritoresReduzidos(todosOsRegistros, quantFramesKmeansPCA);

    descritoresReduzidosCarregados = kmeansFitTeste(centroides, descritoresReduzidosCarregados);


    atualizarCentroides(centroides);
    salvarClassificacoes(centroides, descritoresReduzidosCarregados);

    CentroidesRegistros centroidesRegistros;
    centroidesRegistros.centroides = centroides;
    centroidesRegistros.registros = descritoresReduzidosCarregados;

    return centroidesRegistros;
}

void extrairDescritoresThread(matrix<rgb_pixel> face, std::vector<matrix<float, 0, 1>>& face_descriptors, anet_type extrator_descritor_facial) {

    auto descriptor = extrator_descritor_facial(face);

    std::lock_guard<std::mutex> lock(mtx);
    face_descriptors.push_back(descriptor);
}

void extrairFaceThread(const fs::path& fotos, std::vector<matrix<rgb_pixel>>& faces, frontal_face_detector detector_face, shape_predictor detector_pontos) {

    cv::Mat imagem = cv::imread(fotos.string());
    dlib::array2d<bgr_pixel> dlibImg;


    if (imagem.cols > 800 || imagem.rows > 800) {

        double nova_largura = 800;
        double nova_altura = 0;

        nova_altura = floor((nova_largura * imagem.cols) / imagem.rows);


        cv::Mat imagem_redimensionada;
        cv::resize(imagem, imagem_redimensionada, cv::Size(nova_largura, nova_altura));


        //Converte o Image do OpenCv para o Dlib    
        dlib::assign_image(dlibImg, dlib::cv_image<bgr_pixel>(imagem_redimensionada));
    }
    else {

        //Converte o Image do OpenCv para o Dlib    
        dlib::assign_image(dlibImg, dlib::cv_image<bgr_pixel>(imagem));
    }



    //Os rostos são convertidos em um formato que dê para ser lido pelo descritor
    for (auto face : detector_face(dlibImg)) {

        auto pontos = detector_pontos(dlibImg, face);
        matrix<rgb_pixel> face_chip;
        dlib::extract_image_chip(dlibImg, get_face_chip_details(pontos, 150, 0.25), face_chip);

        auto faceMove = std::move(face_chip);

        std::lock_guard<std::mutex> lock(mtx);
        faces.push_back(faceMove);
    }
}


std::vector<int> fibonnaciSearchAndProcess(std::vector<FuncClust>distancias, int ponteiroInicio, int ponteiroFim, double busca, int ate) {

    std::vector<int>posicoes;
    if (distancias.size() >= 2) {

        if (distancias[0].distancia < busca && distancias[1].distancia > busca) {
            posicoes.push_back(distancias[0].distancia);
            posicoes.push_back(distancias[1].distancia);
        }
        else {

            if (busca > distancias[0].distancia && busca < distancias[distancias.size() - 1].distancia) {

                if (ponteiroFim == 0) {

                    ponteiroInicio = 0;
                    if (ate % 2 == 0) {
                        ponteiroFim = ate / 2;
                    }
                    else {
                        ponteiroFim = (ate + 1) / 2;
                    }
                }
                else {
                    if (ponteiroInicio == 0) {
                        int ponteiroAux = ponteiroFim;
                        ponteiroInicio = ponteiroFim;
                        ponteiroFim = ((ate - ponteiroFim) / 2) + ponteiroFim;
                    }
                    else {

                        if (ponteiroInicio == ponteiroFim) {
                            ponteiroInicio = 0;
                        }

                        int diferenca = ponteiroFim - ponteiroInicio;

                        if (diferenca % 2 == 0) {
                            ponteiroFim = (diferenca / 2) + ponteiroInicio;
                        }
                        else {
                            ponteiroFim = ((diferenca + 1) / 2) + ponteiroInicio;
                        }
                    }
                }

                int ultimaPosicao = 0;

                if (ponteiroFim + 1 > distancias.size() - 1) {
                    ultimaPosicao = distancias.size() - 1;
                }
                else {
                    ultimaPosicao = ponteiroFim + 1;
                }


                //Encontrou valores entre
                if (distancias[ponteiroFim].distancia < busca && distancias[ultimaPosicao].distancia > busca) {

                    posicoes.push_back(ponteiroFim);
                    posicoes.push_back(ultimaPosicao);
                }
                else if (distancias[ponteiroFim].distancia < busca && distancias[ultimaPosicao].distancia < busca) {
                    ponteiroInicio = ponteiroFim;
                    posicoes = fibonnaciSearchAndProcess(distancias, ponteiroInicio, distancias.size() - 1, busca, distancias.size() - 1);
                }
                else if (distancias[ponteiroFim].distancia > busca && distancias[ultimaPosicao].distancia > busca) {
                    posicoes = fibonnaciSearchAndProcess(distancias, ponteiroInicio, ponteiroFim, busca, ponteiroFim);
                }

                //Encontrou valores iguas
                else if (distancias[ponteiroFim].distancia == busca) {
                    posicoes.push_back(ponteiroFim);
                }
                else if (distancias[ultimaPosicao].distancia == busca) {
                    posicoes.push_back(ultimaPosicao);
                }
            }
            else {
                if (busca < distancias[0].distancia) {
                    posicoes.push_back(0);
                }
                else if (busca > distancias[distancias.size() - 1].distancia) {
                    posicoes.push_back(distancias.size() - 1);
                }
            }
        }
    }
    else {        
        posicoes.push_back(distancias[0].distancia);
    }    

    return posicoes;
}

void encontrarPessoa(int posicao,std::vector<FuncClust>funcionarios,bool direcao, std::vector<float>mediasBusca, std::vector<std::string> todosOsCadastros,int idCentroide) {
        
    
    int percentual = ceil(funcionarios.size() * 0.3);
    
    if (direcao) {
        int contador = 0;

        for (int l = posicao; l < funcionarios.size(); l++) {
            
            
            if (!encontrouFuncionario) {

                encontraSimilaridade(funcionarios[l], mediasBusca, todosOsCadastros, idCentroide);
            }
            else {
                l = funcionarios.size();
            }


            if (contador >= percentual) {
                l = funcionarios.size();
            }

            contador += 1;
        }
    }
    else {

        int contador = 0;

        for (int l = posicao; l > 0; l--) {
            
            if (!encontrouFuncionario) {

                encontraSimilaridade(funcionarios[l], mediasBusca, todosOsCadastros, idCentroide);
            }
            else {
                l = 0;
            }

            if (contador >= percentual) {
                l = 0;
            }

            contador += 1;
        }
    }    
}

void encontraSimilaridade(FuncClust funcionario, std::vector<float>mediasBusca, std::vector<std::string> todosOsCadastros, int idCentroide) {

    Registro media = carregarMedias(funcionario.idFunc, "descMediasReduzidas", todosOsCadastros);

    if (media.dimensoes.size() > 0) {

        std::vector <std::vector<float>> vetores;

        vetores.push_back(mediasBusca);
        vetores.push_back(media.dimensoes);

        int tamanhoVetores = confereTamanhoMatriz(vetores);

        double similaridade = similaridaDeCosseno(vetores, tamanhoVetores);

        if (similaridade > 0.8) {

            FuncionarioEncontrado func;            
            func.idCentroide = idCentroide;            
            func.idFuncionario = funcionario.idFunc;
            func.distancia = funcionario.distancia;
            func.similaridade = similaridade;
        
            funcFind = func;
            encontrouFuncionario = true;            
        }
    }
}

std::vector<matrix<float, 0, 1>> extrairDescritores(std::vector<matrix<rgb_pixel>> faces, anet_type extrator_descritor_facial) {

    std::vector<matrix<float, 0, 1>> face_descriptors;
    std::vector<std::thread> threadsExtrairDescritores;

   

    for (const auto& face : faces) {
        threadsExtrairDescritores.push_back(std::thread([&face, &face_descriptors, &extrator_descritor_facial]() {
            extrairDescritoresThread(face, face_descriptors, extrator_descritor_facial);
            }));
    }


    for (auto& t : threadsExtrairDescritores) {
        if (t.joinable()) {
            t.join();
        }
    }

    return face_descriptors;
}

std::vector<matrix<rgb_pixel>> extrairFace(shape_predictor detector_pontos, std::vector<fs::path> image_paths, frontal_face_detector detector_face) {

    std::vector<matrix<rgb_pixel>> faces;
    std::vector<std::thread> threadsExtrairFaces;

    for (const fs::path& path : image_paths) {
        threadsExtrairFaces.push_back(std::thread([&path, &faces, detector_face, detector_pontos]() {
            extrairFaceThread(path, faces, detector_face, detector_pontos);
            }));
    }

    for (auto& t : threadsExtrairFaces) {
        if (t.joinable()) {
            t.join();
        }
    }

    return faces;
}


std::vector<int> possiveisCentroides(std::vector<Centroid>centroides,std::vector<float>dicritoresReduzidos) {

    std::vector<int>centroidesSimilarityCossine;
    for (int j = 0; j < centroides.size(); j++) {

        std::vector<std::vector<float>> vetores;

        vetores.push_back(dicritoresReduzidos);
        vetores.push_back(centroides[j].dimensoes);

        int tamanhoVetores = confereTamanhoMatriz(vetores);

        double similaridade = similaridaDeCosseno(vetores, tamanhoVetores);
        if (similaridade > 0.4) {
            centroidesSimilarityCossine.push_back(centroides[j].id);            
        }

    }

    return centroidesSimilarityCossine;
}


std::vector<int> separaValoresRepetidos(std::vector<std::vector<int>>matriz) {

    std::vector<int>valoresUnicos;
    for (int i = 0; i < matriz.size(); i++) {

        //Percorre primeiro vetor
        for (int j = 0; j < matriz[i].size(); j++) {

            if (i + 1 < matriz.size()) {

                //Percore os demais vetores
                for (int k = i + 1; k < matriz.size(); k++) {

                    for (int l = 0; l < matriz[k].size(); l++) {

                        if (matriz[i][j] == matriz[k][l]) {

                            valoresUnicos.push_back(matriz[i][j]);
                            l = matriz[k].size();
                        }
                    }
                }
            }
        }
    }

    return valoresUnicos;
}
