clear all
%Cargando los datos
%Analisis exploratorio
data=xlsread('COVID19.xlsx','CutAdim');
% plotmatrix(data(:,1:end-2))

%% Deteccion de Outliers
%Cambiar por corte, no 5 peores
d = mahal(data(:,1:end-2),data(:,1:end-2));
dr = mahal_rob(data(:,1:end-2), 'Spearman');
% [m n] =size(data);.
[va id]=maxk(d,5);
[val idx]=maxk(dr,5);
Outliers_Trad=sort(id);
Outliers_Robusta=sort(idx);
Outliers_conCategoricas=table(Outliers_Trad,Outliers_Robusta)

data(idx,:)=[]; %Remover outliers
%% Regresiones
%Multivariado

%REMOVER VARIABLES NO SIGNIFICATIVAS

%Parametrico
X_Explicativos = data(:,2:end-2);
y_casos = data(:,1);
Mod_Parametrico_Mult = fitlm(X_Explicativos,y_casos)
Mod_Parametrico_Mult_Rob = fitlm(X_Explicativos,y_casos,'RobustOpts','on')


%Robusto
[n p] = size(X_Explicativos);

MCovS = zeros(p,p);
MCovK = zeros(p,p);
MCovx_yS = zeros(1,p);
MCovx_yK = zeros(1,p);
corrXS = corr(X_Explicativos, 'Type', 'Spearman');
corrXK = corr(X_Explicativos, 'Type', 'Kendall');    
%Covarianza Robusta
for j=1:p
    MCovx_yS(:,j) = corr(X_Explicativos(:,j),y_casos, 'Type', 'Spearman')*std(X_Explicativos(:,j))*std(y_casos);
    MCovx_yK(:,j) = corr(X_Explicativos(:,j),y_casos, 'Type', 'Kendall')*std(X_Explicativos(:,j))*std(y_casos);
    for i=1:p
        MCovS(i,j) = corrXS(i,j)*std(X_Explicativos(:,i))*std(X_Explicativos(:,j));
        MCovK(i,j) = corrXK(i,j)*std(X_Explicativos(:,i))*std(X_Explicativos(:,j));
    end
end
BetaKMult = pinv(MCovK)*MCovx_yK';
Beta0KMult = mean(y_casos)-BetaKMult'*mean(X_Explicativos)';

BetaSMult = pinv(MCovS)*MCovx_yS';
Beta0SMult = mean(y_casos)-BetaSMult'*mean(X_Explicativos)';

Mod_Kendall_Mult = Beta0KMult+BetaKMult'*X_Explicativos';
Mod_Spearman_Mult = Beta0SMult+BetaSMult'*X_Explicativos';

r2_ParametricoM = Mod_Parametrico_Mult.Rsquared.Adjusted;
r2_ParametricoM_Rob = Mod_Parametrico_Mult_Rob.Rsquared.Adjusted;
r2_KendallM = R2(Mod_Kendall_Mult',y_casos);
r2_SpearmanM = R2(Mod_Spearman_Mult',y_casos);
R2_Reg_Mult=table(r2_ParametricoM,r2_ParametricoM_Rob,r2_KendallM,r2_SpearmanM)


%% Bivariado
a=5;
b=6;
x_explicativa=data(:,a);
y_respuesta=data(:,b);
%Parametrica
Mod_ParametricoBiv = fitlm(x_explicativa,y_respuesta);

%Robusto
Mod_ParametricoBiv_Rob = fitlm(x_explicativa,y_respuesta,'RobustOpts','on');

%Nadaraya
Kernel=2;
[Mod_NadarayaWatson] = nonpr(x_explicativa,y_respuesta,Kernel);

%R2
r2_Mod_Parametrico_Biv = R2(Mod_ParametricoBiv.Fitted,y_respuesta);
r2_Mod_ParametricoBiv_Rob = R2(Mod_ParametricoBiv_Rob.Fitted,y_respuesta);
r2_Nadaraya_Watson = R2(Mod_NadarayaWatson,y_respuesta);
R2_Reg_Biv = table(r2_Mod_Parametrico_Biv,r2_Mod_ParametricoBiv_Rob,r2_Nadaraya_Watson)

%plots
figure
hold on
plot(x_explicativa,y_respuesta,'o','LineWidth',1)
plot(x_explicativa,Mod_ParametricoBiv.Fitted,'.m','LineWidth',1.4)
plot(x_explicativa,Mod_ParametricoBiv_Rob.Fitted,':k','LineWidth',1.3)
[Mod_NadarayaWatson] = nonpr(x_explicativa,y_respuesta,Kernel);
legend('Data',['LinModel R2=',num2str(r2_Mod_Parametrico_Biv)],['RobLinModel R2=',num2str(r2_Mod_ParametricoBiv_Rob)],['Nadaraya-Watson R2=',num2str(r2_Nadaraya_Watson)],'FontSize',13);
title('Personal Medico VS Camas UCI','FontSize',16)
hold off

%% Componentes Principales
datos = data(:,1:end-2);
[n p] = size(datos);
MCovS = zeros(p,p);
corrXS = corr(datos, 'Type', 'Spearman'); 
%Covarianza Robusta
for j=1:p
    for i=1:p
        MCovS(i,j) = corrXS(i,j)*std(datos(:,i))*std(datos(:,j));
    end
end
[vec_rob, Autovalores] = eig(MCovS);
% Componentes principales
Z_rob = zeros(n,p);
for i = 1:p
    Z_rob(:,i) = datos*vec_rob(:,i);
end
VarAcum_robusto=(Autovalores)/sum(Autovalores)
PrimerComp = Z_rob(:,1);

%Componentes principales sin casos
datos_sincasos = data(:,2:end-2);
[n p1] = size(datos_sincasos);
MCovS = zeros(p1,p1);
corrXS = corr(datos_sincasos, 'Type', 'Spearman'); 
%Covarianza Robusta
for j=1:p1
    for i=1:p1
        MCovS(i,j) = corrXS(i,j)*std(datos_sincasos(:,i))*std(datos_sincasos(:,j));
    end
end
[vec_rob, Autovalores] = eig(MCovS);
% Componentes principales
Z_trad_sincasos = zeros(n,p1);
Z_rob_sincasos = zeros(n,p1);
for i = 1:p1
    Z_rob_sincasos(:,i) = datos_sincasos*vec_rob(:,i);
end
VarAcum_robusto_sincasos=(Autovalores)/sum(Autovalores)
PrimerComp = Z_rob_sincasos(:,1);
Y = data(:,1);
Mod_Lineal = fitlm(Z_rob_sincasos(:,1:3),Y);




%% Clasificacion
X_Explicativos = data(:,1:end-2);
Label_Preparacion = data(:,end-1);
Label_Region = data(:,end);

Clasif_kmeans = kmeans(X_Explicativos,2);
MC_kmeans =  confusionmat(Label,Clasif_kmeans)

Clasif_kmedoids = kmedoids(X_Explicativos,2);
MC_kmedoids =  confusionmat(Label,Clasif_kmedoids)

[moda, n] = Fstatistic(X_Explicativos)

zzzz=kmeans(PrimerComp,2);
Mc=confusionmat(Label,zzzz)
Acc = trace(Mc)/sum(sum(Mc))

%Clasificacion con k medias no funciona, muy variante, probar con
%crossvalidation
%REGRESION LOGISTICA 
%CROSS VALIDATION CON SUS RESPECTIVOS ESTADISTICOS, F1 SCORE, ACCURACY ....












