%Subo la base de datos
load('BDProyecto.mat')
BD=BDProyecto;

%Variables explicativas
X=BD(:,1:end-1);

%Variable respuesta
Y=BD(:,end)-1;
 
%Separo en train y test
%Para hacerlo de forma aleatoria hago una permutacion aleatoria
for h = 1:100
n=length(BD);
idx = randperm(n);
X_train = X(idx(1:floor(n*0.7)),:);
y_train = Y(idx(1:floor(n*0.7)));
X_test = X(idx(floor(n*0.7+1):end),:);
y_test = Y(idx(floor(n*0.7+1):end));
S = LogitRegression2(X_train,X_test,y_train,y_test,0);
J(h)=S.test_acc;
end
plot(J)
median(J)
function S = LogitRegression2(X_train,X_test,y_train,y_test,plot)
S = struct();
%Con la parte de entrenamiento hago el ajuste a un modelo logistico
modelo=fitglm(X_train,y_train,'Distribution','binomial','link','logit');

%Miro el p-valor para ver la significancia (menos el del intercepto)
pvalor=table2array(modelo.Coefficients(2:end,4));

%Elimino progresivamente el mayor p-value hasta que todos sean <0.05
while max(pvalor)>=0.05
    maxp=find(pvalor==max(pvalor));
    pvalor(maxp)=[];
    X_train(:,maxp)=[];
    X_test(:,maxp)=[];
    modelo=fitglm(X_train,y_train,'Distribution','binomial','link','logit');
    pvalor=table2array(modelo.Coefficients(2:end,4));
    intercepto=table2array(modelo.Coefficients(1,4));
end

%Para ver los coeficientes del modelo
S.CoefMod = table2array(modelo.Coefficients(:,1));

%Predigo los resultados, como es logistic regresion la respuesta son
%probabilidades
ypred = predict(modelo,X_test);
S.y_test_pred=[];

%Convierto las probabilidades en 0 y 1
for j=1:length(ypred)
   if ypred(j)>=0.45    %Puedo cambiar la probabilidad de corte
       S.y_test_pred(j)=1;     %Estos valores serán según las clases
   else S.y_test_pred(j)=0;
   end
end

%Matriz de confusión
CM=confusionmat(y_test,S.y_test_pred);
S.test_acc=trace(CM)/sum(sum(CM));

%Para saber el accuracy de train hago lo mismo que hice con test
ytpred=predict(modelo,X_train);
S.y_train_pred=[];
for j=1:length(ytpred)
   if ytpred(j)>=0.45       %Acá también debo cambiarla
       S.y_train_pred(j)=1;
   else S.y_train_pred(j)=0;
   end
end

%Confusion matrix normal
CM2=confusionmat(y_train,S.y_train_pred);
S.train_acc=trace(CM2)/sum(sum(CM2));

if (plot==1)
% Test plot
figure()
confusionchart(y_test,S.y_test_pred)
title('Confusion Matrix Test')

% Train plot
figure()
confusionchart(y_train,S.y_train_pred);
title('Confusion Matrix Train')

%Comparo el accuracy de train y test
S.test_acc;
S.train_acc;
else 
end
end


