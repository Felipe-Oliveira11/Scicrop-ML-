import numpy as np
import pandas as pd
import pickle
import streamlit as st


print('Pandas {}'.format(pd.__version__))
print('Numpy {}'.format(np.__version__))
print('Streamlit {}'.format(st.__version__))
class_names = ['saudável', 'pesticidas', 'outros']


# load model
load_model = pickle.load(
    open('C:\\Users\\Marcello\\Desktop\\SciCrop\\xgb_clf.pkl', 'rb'))


# main menu
def menu():

    st.sidebar.header('Home')
    page = st.sidebar.radio("", ('SciCrop',
                                 'ML Plataform',
                                 'Safra Predição',
                                 'Sobre'))
    # hide the menu
    hide_streamlit_style = """
                <style>
                # MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    if page == 'SciCrop':
        scicrop()

    if page == 'ML Plataform':
        plataform()

    if page == 'Safra Predição':
        safra()

    if page == 'Sobre':
        contato()


# ScriCrop page
def scicrop():
    st.title('SciCrop')
    st.write('A SciCrop é uma startup no segmento de soluções de Big Data e hoje oferece soluções de processamento de dados mais rápidas, eficientes e flexíveis para fornecer informações estratégicas por meio de algoritmos para a gestão de seus negócios.')
    st.image(
        'https://agevolution.canalrural.com.br/wp-content/uploads/2019/06/Scicrop.jpg',
        width=700,
        height=700,
        caption='SciCrop Logo')


# SciCrop ML
def plataform():
    st.title('Machine learning Plataform')
    st.write('Apesar do agro gerar muito lucro, a vida dos agricultores não é fácil, mas sim um verdadeiro teste de resistência e determinação. Uma vez que tenhamos semeado as sementes, o agricultor precisa trabalhar dia e noite para garantir uma boa safra no final da estação. Uma boa colheita depende de diversos fatores \
             como disponibilidade de água, fertilidade do solo, proteção das culturas, uso oportuno de pesticidas, outros fatores químicos úteis e da natureza. ​Muitos desses dados são quase impossíveis de se controlar, mas a quantidade e a frequência de pesticidas é algo que o agricultor pode administrar. Os pesticidas podem protegem a colheita com a dosagem certa. Mas, se adicionados em quantidade inadequada, podem prejudicar toda a safra.')
    st.write('\n')
    st.image('https://www.sulinfoco.com.br/wp-content/uploads/2019/05/Agricultura-Giro-Rural-1.jpg',
             width=600,
             height=300)
    st.write('\n')
    st.write('Essa plataforma de Machine learning foi desenvolvida para resolver está dor do Agricultor brasileiro, na plataforma ele consegue fazer a previsão da colheita da Safra.  \
             A plataforma é armazenada na nuvem e está disponível para uso do agricultor, permitindo que ele tenha acesso apenas entrando neste aplicativo da web e preenchendo os dados, \
             em segundos a inferência é feita no modelo de aprendizado de máquina e ele tem a previsão em mãos.')


# Safra Prediction
def safra():
    st.image(
        'https://agevolution.canalrural.com.br/wp-content/uploads/2019/06/Scicrop.jpg',
        width=500,
        height=200)
    st.title('Safra Prediction')
    st.write('\n')
    st.write('\n')
    st.write('\n')

    # estimativa de insetos
    numero_insetos = st.number_input(
        'Estimativa_de_Insetos', min_value=0, max_value=100000, value=0)

    # tipo de cultivo
    tipo_cultivo = st.selectbox('Tipo_de_Cultivo', [0, 1])

    # tipo de solo
    tipo_solo = st.selectbox('Tipo_de_Solo', [0, 1])

    # categoria pesticidas
    cat_pesticidas = st.selectbox('Categoria_Pesticida', [0, 1, 2])

    # doses semana
    doses_semana = st.number_input(
        'Doses_Semana', min_value=0, max_value=10000000, value=0)

    # semanas utilizando
    semanas_utlizando = st.number_input(
        'Semanas_Utilizando', min_value=0, max_value=10000000, value=0)
    # semanas sem uso
    semanas_sem_uso = st.number_input(
        'Semanas_Sem_Uso', min_value=0, max_value=1000000, value=0)

    # temporada climática
    temporada = st.selectbox('Temporada', [1, 2, 3])

    input_dict = {'Estimativa_de_Insetos': [numero_insetos], 'Tipo_de_Cultivo': [tipo_cultivo],
                  'Tipo_de_Solo': [tipo_solo], 'Categoria_Pesticida': [cat_pesticidas],
                  'Doses_Semana': [doses_semana], 'Semanas_Utilizando': [semanas_utlizando],
                  'Semanas_Sem_Uso': [semanas_sem_uso], 'Temporada': [temporada]}

    input_df = pd.DataFrame(input_dict)
    load_model = pickle.load(open('xgb_clf.pkl', 'rb'))

    if st.button('Predição'):
        output = load_model.predict(input_df)
        if output == 0:
            st.success(' Previsão Colheita de Safra:  {}'.format('Sem danos'))
        elif output == 1:
            st.success(' Previsão Colheita de Safra:  {}'.format(
                'Danos por outros motivos'))
        else:
            st.success('Previsão Colheita de Safra:  {}'.format(
                'Danos gerados por pesticidas'))


# contato
def contato():
    st.title('Contato')
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSh8_BbTxZSsHWdLsSVjvVGVjASl3WynpmbMg&usqp=CAU',
             width=100, height=100)
    st.write('\n')
    st.write('\n')
    st.write('Este projeto foi desenvolvido por Felipe Oliveira, \
             Dúvidas ou sugestões podem me encaminhar um e-mail ou uma mensagem no LinkedIn.')
    st.write('\n')
    st.markdown(
        '[LinkedIn](https://www.linkedin.com/in/felipe-oliveira-18a573189/)')
    st.write('\n')
    st.write('E-mail: felipe.oliveiras2000@gmail.com')


if __name__ == '__main__':
    menu()
