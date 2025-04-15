#######################################################################
#######################################################################
# App de Comanda de restaurant (+ Analitica)
#######################################################################
#######################################################################


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [A] Importacion de librerias
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# Obtener versiones de paquetes instalados
# !pip list > requirements.txt

import pandas as pd
import numpy as np

import math
from datetime import datetime
import pytz


# para tratamiento de archivos
import zipfile
from io import BytesIO

import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image

import streamlit as st

import warnings
warnings.filterwarnings('ignore')








#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [B] Creacion de funciones internas utiles
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


#=======================================================================
# [B.1] Graficar imagenes en grilla 
#=======================================================================


@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def mostrar_grilla_imagenes(
  dic_imgs,
  fsize = 3  
  ):
  n = len(dic_imgs)
  if n == 0:
    return None

  # Calcular filas y columnas para una grilla lo más cuadrada posible
  cols = math.ceil(math.sqrt(n))
  rows = math.ceil(n / cols)

  fig, axs = plt.subplots(rows, cols, figsize=(cols * fsize, rows * fsize))
  axs = axs.flatten()

  for ax in axs[n:]:
    ax.axis('off')  # Desactivar los ejes vacíos al final

  for i, (nombre, contenido) in enumerate(dic_imgs.items()):
    img = Image.open(BytesIO(contenido))
    axs[i].imshow(img)
    axs[i].set_title(nombre, fontsize=9)
    axs[i].axis('off')

  plt.tight_layout()
  return fig


#=======================================================================
# [B.2] Funcion de Obtener hora de ahora con horario santiago de chile
#=======================================================================


def ahora():
  santiago_tz = pytz.timezone('America/Santiago')
  ahora_naive = datetime.now() # Obtiene la hora local del servidor (puede ser UTC)
  ahora_santiago = santiago_tz.localize(ahora_naive)
  return ahora_santiago.strftime('%Y-%m-%d %H:%M:%S')


#=======================================================================
# [B.3] Entregables de analisis
#=======================================================================


@st.cache_resource()
def analisis_menu(
  df_ventas,
  df_menu,
  df_ingredientes,
  fecha_desde,
  fecha_hasta
  ):  
  
  #________________________________________________
  # Trabajar df
  
  df_ventas2 = df_ventas[
    df_ventas['estado']=='cerrada'
    ].groupby([
      'hora cierre',
      'mesa',
      'id',
      'plato'  
      ]).agg( 
        hora_llegada = pd.NamedAgg(column = 'hora pedido', aggfunc = min),
        venta = pd.NamedAgg(column='precio', aggfunc = sum),
        pedidos = pd.NamedAgg(
          column='accion', 
          aggfunc= lambda x: (x=='Agregar').sum() - (x=='Quitar').sum()
          )
        ).reset_index()
  df_ventas2 = df_ventas2[df_ventas2['pedidos']>0]

  df_ventas2['hora cierre'] = pd.to_datetime(df_ventas2['hora cierre'])
  df_ventas2['hora_llegada'] = pd.to_datetime(df_ventas2['hora_llegada'])

  df_ventas3 = df_ventas2[
    (df_ventas2['hora cierre'].dt.date>=fecha_desde) & 
    (df_ventas2['hora cierre'].dt.date<=fecha_hasta)
  ]


  dic_dia_semana = {
    'Monday': 'Lunes',
    'Tuesday': 'Martes',
    'Wednesday': 'Miercoles',
    'Thursday': 'Jueves',
    'Friday': 'Viernes',
    'Saturday': 'Sabado',
    'Sunday': 'Domingo'
    }

  df_ventas3['dia_semana'] = df_ventas3['hora_llegada'].apply(
    lambda x: dic_dia_semana[x.day_name()]
    )

  df_ventas3['hora'] = df_ventas3['hora_llegada'].dt.hour
  df_ventas3['fecha'] = df_ventas3['hora_llegada'].dt.strftime('%d-%m-%Y')

  df_ventas3 = df_ventas3.merge(
    df_menu[['Id','Categoria']],
    how='left',
    right_on='Id',
    left_on='id'
    )


  #________________________________________________
  # Ver participacion por categoria (treemap)

  df_ventas3_pc = df_ventas3.groupby([
    'Categoria',
    'plato'
    ]).agg( 
      venta = pd.NamedAgg(column='venta', aggfunc = sum),
      pedidos = pd.NamedAgg(column='pedidos', aggfunc = sum)
      ).reset_index()

  df_ventas3_pc['plato'] = df_ventas3_pc['plato'].apply(
    lambda x: x.replace(' ','<br>')
    )

  fig_pc = px.treemap(
    df_ventas3_pc, 
    path=[px.Constant('Menu'), 'Categoria', 'plato'], 
    values='venta',
    color='Categoria', 
    hover_data=['venta','pedidos']
    )
  fig_pc.update_layout(margin = dict(t=50, l=25, r=25, b=25))



  #________________________________________________
  # Ver participacion por dia de la semana

  df_ventas3_ds = df_ventas3.groupby([
    'dia_semana',
    'Categoria',
    'plato'
    ]).agg( 
      venta = pd.NamedAgg(column='venta', aggfunc = sum),
      pedidos = pd.NamedAgg(column='pedidos', aggfunc = sum)
      ).reset_index()
  
  df_ventas3_ds['total_venta_dia'] = df_ventas3_ds.groupby('dia_semana')['venta'].transform('sum')
  df_ventas3_ds['peso'] = (df_ventas3_ds['venta'] / df_ventas3_ds['total_venta_dia']) * 100
  
  df_ventas3_ds['plato'] = df_ventas3_ds.apply(
    lambda x: x['Categoria']+': '+x['plato'],
    axis=1
  )
  
  fig_ds = px.area(
    df_ventas3_ds,
    x='dia_semana',
    y='peso',
    color='plato',
    category_orders={
      'dia_semana': ['Lunes','Martes','Miercoles','Jueves','Viernes','Sabado','Domingo']
      }
    )
  fig_ds.update_layout(yaxis_range=[0, 100])



  #________________________________________________
  # Ver participacion por hora del dia
      

  df_ventas3_h = df_ventas3.groupby([
    'hora',
    'Categoria',
    'plato'
    ]).agg( 
      venta = pd.NamedAgg(column='venta', aggfunc = sum),
      pedidos = pd.NamedAgg(column='pedidos', aggfunc = sum)
      ).reset_index()
  
  df_ventas3_h['total_venta_hora'] = df_ventas3_h.groupby('hora')['venta'].transform('sum')
  df_ventas3_h['peso'] = (df_ventas3_h['venta'] / df_ventas3_h['total_venta_hora']) * 100
  
  df_ventas3_h['plato'] = df_ventas3_h.apply(
    lambda x: x['Categoria']+': '+x['plato'],
    axis=1
  )

  df_ventas3_h['hora'] = df_ventas3_h['hora'].astype('str')

  fig_h = px.area(
    df_ventas3_h,
    x='hora',
    y='peso',
    color='plato'
    )
  fig_h.update_layout(yaxis_range=[0, 100])
  
  
  #________________________________________________
  # Ver evolucion de ventas en el tiempo

  df_ventas3_t = df_ventas3.groupby([
    'fecha',
    'Categoria',
    'plato'
    ]).agg( 
      venta = pd.NamedAgg(column='venta', aggfunc = sum),
      pedidos = pd.NamedAgg(column='pedidos', aggfunc = sum)
      ).reset_index()

  df_ventas3_t['fecha'] = pd.to_datetime(df_ventas3_t['fecha'], format='%d-%m-%Y')

  df_ventas3_t['plato'] = df_ventas3_t.apply(
    lambda x: x['Categoria']+': '+x['plato'],
    axis=1
  )

  df_ventas3_t = df_ventas3_t.sort_values(by=['plato','fecha'])
  
  fig_t = px.line(
    df_ventas3_t,
    x='fecha',
    y='venta',
    color='plato'
    )

  #________________________________________________
  # Ver histograma de lo que tarda una mesa



  df_ventas3_h = df_ventas3.groupby([
    'hora cierre',
    'mesa'
    ]).agg( 
      hora_llegada = pd.NamedAgg(column='hora_llegada', aggfunc = min)
      ).reset_index()
  
  df_ventas3_h['hora_llegada'] = pd.to_datetime(df_ventas3_h['hora_llegada'])
  df_ventas3_h['hora cierre'] = pd.to_datetime(df_ventas3_h['hora cierre'])
  df_ventas3_h['diferencia_minutos'] = (df_ventas3_h['hora cierre'] - df_ventas3_h['hora_llegada']).dt.total_seconds() / 60
  
  fig_hist = px.histogram(
    df_ventas3_h, 
    x='diferencia_minutos',
    title='Tiempo Mesa',
    marginal='box'  
    )



  #________________________________________________
  # Ver violin de hora de llegada por dia


  df_ventas3_hll = df_ventas3.groupby([
    'dia_semana',
    'hora cierre',
    'mesa'
    ]).agg( 
      hora = pd.NamedAgg(column='hora', aggfunc = min)
      ).reset_index()


  fig_hll = px.violin(
    df_ventas3_hll, 
    x='dia_semana',
    y='hora',
    title='Distribucion segun hora de llegada y dia',
    category_orders={
      'dia_semana': ['Lunes','Martes','Miercoles','Jueves','Viernes','Sabado','Domingo']
      }
    )

  #________________________________________________
  # Ver evolucion de consumo de ingredientes 
  
  

  df_ventas3_i = df_ventas3.groupby(['fecha','dia_semana','Id']).agg( 
    pedidos = pd.NamedAgg(column='pedidos', aggfunc = sum)
    ).reset_index().merge(
    df_ingredientes,
    how = 'left',
    on='Id'
  )    

  df_ventas3_i['Cantidad'] = df_ventas3_i.apply(
    lambda x: float(x['pedidos'])*float(x['Cantidad'].replace(',','.')),
    axis=1
  )

  df_ventas3_i = df_ventas3_i.groupby(['fecha','dia_semana','Ingrediente','Unidad']).agg( 
    Cantidad = pd.NamedAgg(column='Cantidad', aggfunc = sum)
    ).reset_index()

  df_ventas3_i['Ingrediente'] = df_ventas3_i.apply(
    lambda x: x['Ingrediente']+' ['+x['Unidad']+']',
    axis=1
  )

  df_ventas3_i['fecha2'] = pd.to_datetime(df_ventas3_i['fecha'], format='%d-%m-%Y')
  
  df_ventas3_i = df_ventas3_i.sort_values(
    by=['Ingrediente','fecha2']
    ).reset_index(drop=True)

  df_ventas3_i['Cantidad_Acumulada'] = df_ventas3_i.groupby('Ingrediente')['Cantidad'].cumsum()
  
  df_ventas3_i2 = df_ventas3_i.pivot_table(
    index=['fecha2','fecha','dia_semana'],
    columns='Ingrediente',
    values='Cantidad_Acumulada'
  ).droplevel('fecha2')



  #________________________________________________
  # Generar entregables
  
  return fig_pc,fig_ds,fig_h,fig_t,fig_hist,fig_hll,df_ventas3_i2
  


#=======================================================================
# [B.2] Desplegar una imagen
#=======================================================================


@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def cargar_imagen(
  dic_imgs,
  id
  ):
  img = Image.open(BytesIO(dic_imgs[f'{id}.jpg']))
  return img



#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [C] Generacion de la App
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
# https://emojiterra.com/es/search/

st.set_page_config(layout='wide')

# titulo inicial 
st.markdown('# 🍽️ Gestion de pedidos y analisis restaurante')



# autoria 
st.sidebar.markdown('**Autor 👉 [Sebastian Barrera](https://www.linkedin.com/in/sebasti%C3%A1n-nicolas-barrera-varas-70699a28)**')


col1, col2 = st.sidebar.columns(2)


archivo_usar = col1.radio(
  'Selecciona una opción:',
  options=['Usar archivo de ejemplo', 'Cargar archivo']
  )

col2.write(' ')
col2.write(' ')
boton_cargar_archivos = col2.button('Cargar Archivos',type='primary')

archivo = None
if archivo_usar == 'Cargar archivo':
  
  archivo = st.sidebar.file_uploader(
    'Sube un archivo .zip',
    type=['zip']
    )

  
  st.sidebar.write('''
    Debes subir un archivo zip que contenga:
    1. archivo 'tabla_menu.csv' con las columnas:\n
    - Id: entero correlativo de cada plato\n
    - Categoria: texto de la categoria del plato (ej: 'postre')\n
    - Nombre: texto con el nombre del plato\n
    - Descripcion: texto con descripcion del plato\n
    - Precio: cifra del precio del plato\n
    2. archivo 'tabla_ingredientes.csv' con las columnas:\n
    - Id: entero correlativo de cada plato\n	
    - Nombre: texto con el nombre del plato\n
    - Ingrediente: texto con ingrediente del plato\n	
    - Cantidad: cifra de cantidad del ingrediente\n
    - Unidad: texto que indica unidad del ingrediente\n
    3. Carpeta llamada 'Imagenes_menu' con imagenes de cada plato
    en formato jpg llamados por su Id (ej: '3.jpg')
    ''')

else:
  
  # cargar archivo local en mismo formato BytesIO que si se hubiese subido por el uploader
  archivo = BytesIO(open('Material_menu.zip','rb').read())
  
  if 'dft' not in st.session_state:
    st.session_state['dft'] = pd.read_csv(
      'tabla_ventas_simulada.csv',
      sep=';',
      decimal=',',
      encoding='utf-8-sig'
      )
      


# Crear tres tabs
tab1,tab2,tab3,tab4,tab5 = st.tabs([
  '📋 Data Cargada', 
  '🍝 Menu',
  '🧾 Cuenta', 
  '📅 Bitacora Pedidos', 
  '📈 Analisis',
  ])



# Inicializamos contenedores en session_state
if 'dfs' not in st.session_state:
  st.session_state['dfs'] = {}

if 'imgs' not in st.session_state:
  st.session_state['imgs'] = {}


# Crear un DataFrame vacío en session_state si no existe
if 'dfp' not in st.session_state:
  st.session_state['dfp'] = pd.DataFrame(columns=['accion','id','plato','precio'])
  
  
# Crear un DataFrame vacío en session_state si no existe
if 'dft' not in st.session_state:
  st.session_state['dft'] = pd.DataFrame(columns=[
    'hora pedido','mesa','accion','id','plato','precio','estado','hora cierre'
    ])


# al presionar boton descomprimir y guardar archivos
if boton_cargar_archivos:
  
  with zipfile.ZipFile(archivo) as zip_ref:
    for nombre in zip_ref.namelist():
      if nombre.endswith('tabla_menu.csv') or nombre.endswith('tabla_ingredientes.csv'):
        with zip_ref.open(nombre) as f:
          df = pd.read_csv(f,sep=';')
          clave = nombre.split('/')[-1].replace('.csv', '')
          st.session_state['dfs'][clave] = df

      elif nombre.startswith('Imagenes_menu/') and nombre.endswith('.jpg'):
        with zip_ref.open(nombre) as f:
          clave = nombre.split('/')[-1]  # Nombre del archivo sin ruta
          st.session_state['imgs'][clave] = f.read()


#_____________________________________________________________________________
# 1.Data Cargada 
    

with tab1:   
  
  if st.session_state['dfs'] is not None:   
  
    with st.expander('Tabla: "tabla_menu.csv"', expanded=False):
      st.dataframe(st.session_state['dfs']['tabla_menu'],hide_index=True)      

    with st.expander('Tabla: "tabla_ingredientes.csv"', expanded=False):
      st.dataframe(st.session_state['dfs']['tabla_ingredientes'],hide_index=True)

    with st.expander('Imagenes en carpera: "Imagenes_menu"', expanded=False):
      fig_tab1 = mostrar_grilla_imagenes(
        dic_imgs = st.session_state['imgs'],
        fsize = 2.5 
        )
      st.pyplot(fig_tab1)
  
  
#_____________________________________________________________________________
# 2. Toma de orden (Menu)


with tab2:  
  
  if 'dfs' in st.session_state:  
    
    dfm = st.session_state['dfs']['tabla_menu']
    
    categorias = dfm['Categoria'].unique().tolist()
    emoticones_cats = ['🥘','🍳','🥙','🍛','🥗','🍖','🍟','🍱','🍔','🍕','🥟']
    
    # definir lista de expanders, boton comprar, boton_reversa
    exps = {}
    b_comprar = {}
    b_reversar = {}
    
    # definir columnas 
    col21, col22 = st.columns([3, 2])
    
    
    for i in range(len(categorias)):
      
      categoria = categorias[i]
      emoji = emoticones_cats[i]
      
      # calcular ids de platos de categoria
      ids = sorted(dfm.loc[dfm['Categoria']==categoria,'Id'].unique().tolist())
      
      exps[categoria] = col21.expander(
        emoji+' '+categoria+' (x'+str(len(ids))+')', 
        expanded=False
        )
            
      # recorrer cada id e ir poblando el expander
      for j in range(len(ids)):
        
        id = ids[j]
        
        nom_plato = dfm.loc[dfm['Id']==id,'Nombre'].item()
        desc_plato = dfm.loc[dfm['Id']==id,'Descripcion'].item()
        precio_plato = dfm.loc[dfm['Id']==id,'Precio'].item()
        
        img = cargar_imagen(          
          dic_imgs = st.session_state['imgs'],
          id = id
          )
        

        col_a,col_b,col_c,col_d = exps[categoria].columns([3,3,1,1])

        col_a.image(img, use_column_width=True)
        col_b.write(nom_plato+' [$'+format(precio_plato,',').replace(',','.')+']')
        col_b.write(desc_plato)
        
        b_comprar[id] = col_c.button('Agregar',type='primary', key=100+id)
        b_reversar[id] = col_d.button('Quitar',key=500+id)
      
          
    # recorrer botones de agregar o reversar para insertar en df de pedido
    for k in sorted(dfm['Id'].unique().tolist()):
      
      nom_plato = dfm.loc[dfm['Id']==k,'Nombre'].item()
      precio_plato = dfm.loc[dfm['Id']==k,'Precio'].item()
      
      if b_comprar[k]:
        
        nuevo_registro = {
          'accion': 'Agregar',
          'id': k,
          'plato': nom_plato,
          'precio': precio_plato
          }
        
        st.session_state['dfp'] = st.session_state['dfp'].append(
          nuevo_registro, 
          ignore_index=True
          )
       
      if b_reversar[k]:
        
        nuevo_registro = {
          'accion': 'Quitar',
          'id': k,
          'plato': nom_plato,
          'precio': -precio_plato
          }
        
        st.session_state['dfp'] = st.session_state['dfp'].append(
          nuevo_registro, 
          ignore_index=True
          )
        
    # agregar widget de mesa y boton de ingresar pedido
    scol1,scol2,scol3 = col22.columns([1,1,1])
    
    mesa = scol1.selectbox('Mesa',range(1, 100),key=705)
        
    scol2.write(' ')
    scol2.write(' ')
    boton_limpiar_pedido = scol2.button('Limpiar',key=701)
    
    if boton_limpiar_pedido:
      st.session_state['dfp'] = pd.DataFrame(columns=['accion','id','plato','precio'])
    
    
    # mostrar boton de ingresar pedido 
    total_pedido = sum(st.session_state['dfp']['precio'])    
    scol3.write(' ')
    scol3.write(' ')
    boton_ingresar_pedido = scol3.button(
      'Ingresar ['+format(total_pedido,',').replace(',','.')+']',
      type='primary',
      key=702
      )
               
    
    # si se presiona el boton de pedido, reiniciar el df_pedido e insertar en historico
    if boton_ingresar_pedido:
      
      df_insertar = st.session_state['dfp'].copy()
      df_insertar['hora pedido'] = ahora()
      df_insertar['mesa'] = mesa
      df_insertar['estado'] = 'abierta'     
      df_insertar['hora cierre'] = ''  
      
      st.session_state['dft'] = pd.concat([
        df_insertar[['hora pedido','mesa','accion','id','plato','precio','estado','hora cierre']],
        st.session_state['dft']        
        ])

      # NO SE ESTA LIMPIANDO EL DF POR ALGUN MOTIVO 
      st.session_state['dfp'] = pd.DataFrame(columns=['accion','id','plato','precio'])
    
  
    
    
    # mostrar df
    col22.dataframe(st.session_state['dfp'],hide_index=True,use_container_width=True)
    
    


#_____________________________________________________________________________
# 3. Cerrar Cuenta
    

with tab3:  
  
  # agregar widget de mesa y boton de ingresar pedido
  col31,col32,col33 = st.columns([1,1,1])
  
  
  # seleccionar mesa para consultar cuenta
  mesa_cuenta = col31.selectbox('Mesa',range(1, 100),key=708)
  
  # rescatar df de esa mesa
  df_cuenta = st.session_state['dft'].copy()
  df_cuenta = df_cuenta.loc[
    (df_cuenta['estado']=='abierta') & (df_cuenta['mesa']==mesa_cuenta),
    ['accion','id','plato','precio']
    ]
  
    
  # mostrar total de la cuenta 
  total_cuenta =  sum(df_cuenta['precio'])
  col32.write(' ')
  col32.write(' ')
  col32.markdown('**'+format(total_cuenta,',').replace(',','.')+'**')
  
  # mostrar boton de cerrar cuenta
  col33.write(' ')
  col33.write(' ')
  boton_cerrar_cuenta = col33.button(
    'Cerrar Cuenta',
    type='primary',
    key=709
    )
    
  # actualizar tabla historica cuando se cierra la cuenta 
  if boton_cerrar_cuenta:    
    
    st.session_state['dft'].loc[
      (st.session_state['dft']['estado']=='abierta') & 
      (st.session_state['dft']['mesa']==mesa_cuenta), 
      ['estado', 'hora cierre']
      ] = ['cerrada',ahora()]


  # mostrar df de cuenta
  st.dataframe(df_cuenta,hide_index=True,use_container_width=True)

#_____________________________________________________________________________
# 4. df en bruto de totalidad de ordenes ingresadas y su estado
 

with tab4:  
  
  st.dataframe(st.session_state['dft'],hide_index=True,use_container_width=True)
  



#_____________________________________________________________________________
# 5. Analisis de venta, inventario, etc


with tab5:  
    
  if len(st.session_state['dft'])>0:
    
    fecha_i = min(pd.to_datetime(st.session_state['dft']['hora pedido']).dt.date)
    fecha_f = max(pd.to_datetime(st.session_state['dft']['hora pedido']).dt.date)

    rango_fechas = st.date_input(
      'Seleccionar rango de fechas',
      (fecha_i,fecha_f),
      fecha_i,
      fecha_f,
      format='DD/MM/YYYY',
      )    
  
    # generar entregables de funcion de analisis
    fig_pc,fig_ds,fig_h,fig_t,fig_hist,fig_hll,df_ingr = analisis_menu(
      df_ventas = st.session_state['dft'],
      df_menu = st.session_state['dfs']['tabla_menu'],
      df_ingredientes = st.session_state['dfs']['tabla_ingredientes'],
      fecha_desde = rango_fechas[0],
      fecha_hasta = rango_fechas[1]
      )
    
    # mostrar resultados de analisis en expander       
    with st.expander('Distribucion Ventas Totales', expanded=False):
      st.plotly_chart(fig_pc)
      
    with st.expander('Distribucion Ventas por dia de semana', expanded=False):
      st.plotly_chart(fig_ds)      
      
    with st.expander('Distribucion Ventas por hora del dia', expanded=False):
      st.plotly_chart(fig_h)         
      
    with st.expander('Evolucion de ventas', expanded=False):
      st.plotly_chart(fig_t)   
      
    with st.expander('Distribucion tiempo de cada mesa', expanded=False):
      st.plotly_chart(fig_hist)   
      
    with st.expander('Distribucion llegadas por dia', expanded=False):
      st.plotly_chart(fig_hll)   
      
    with st.expander('Consumo acumulado de ingredientes', expanded=False):
      st.dataframe(df_ingr,hide_index=False)


  







# !streamlit run App_Comanda_Restaurant4.py

# para obtener TODOS los requerimientos de librerias que se usan
# !pip freeze > requirements.txt

# Video tutorial para deployar una app en streamlitcloud
# https://www.youtube.com/watch?v=HKoOBiAaHGg&ab_channel=Streamlit



