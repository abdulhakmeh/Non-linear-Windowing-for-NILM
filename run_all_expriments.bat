Set _InputFile= E:\ahak15_Projekt\Expriments_parameter\Machine_0_parameters.csv
Set _EnvActivat=C:\Users\ahak15\Anaconda3\Scripts\activate.bat tensorflow-gpu
Set _EnvPath=C:\anaconda3\envs
Set _MainPath=E:\ahak15_Projekt\main.py
FOR /F "tokens=1-8* delims=," %%A IN (%_InputFile%) DO (
Set _ID=%%A
Set _algo=%%B
Set _spacing=%%C
Set _sampling=%%D
Set _value=%%E
Set _linearFlag=%%F
Set _factor=%%G
Set _status=%%H
Set _NumofMachine=0
CALL :PROCESS
)
GOTO :EOF

:PROCESS
IF "%_status%" == "pending" (	
	%_EnvActivat% & cd %_EnvPath% & python %_MainPath% %_ID% %_algo% %_spacing% %_sampling% %_value% %_factor% %_linearFlag% %_NumofMachine%
	)
GOT:EOF