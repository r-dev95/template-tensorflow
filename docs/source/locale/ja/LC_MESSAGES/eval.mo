Þ                        ì  !   í  )     ¸   9  0   ò     #     8  :   S  
             ¡     µ     É  5   Ú       4   '  '   \       I     ,   æ  4     
   H     S  $   q          ¶     Ï     Û     ç    ì  %   m  7     á   Ë  <   ­     ê  !   		  Q   +	     }	  	   	  !   	  !   ¹	     Û	  N   ô	  $   C
  K   h
  <   ´
     ñ
  O     *   ]  W        à     ó  1     !   E  '   g          ¢     ²   **dict[str, Any]** -- parameters. Checks the :class:`Evaluator` parameters. Command line arguments are overridden by file parameters. This means that if you want to set everything using file parameters, you don't necessarily need to use command line arguments. Customize the evaluation of your trained models. Evaluates the model. Loads the evaluation data. Other necessary parameters are set in the file parameters. Parameters Returns Run ``.compile()``. Run ``.summary()``. Runs evaluation. Set only common parameters as command line arguments. Set the model weights. Sets the command line arguments and file parameters. Sets the loss function, model, metrics. Sets up the evaluation. This function is decorated by ``@save_params_log`` and ``@process_time``. This is the module that evaluates the model. Use a yaml file. (:func:`lib.common.file.load_yaml`) class list key=loss: loss function class key=metrics: list of metrics classes key=opt: optimizer method class list of callback classes model class parameters. type Project-Id-Version: template-tensorflow 
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2024-11-23 17:03+0900
PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: ja
Language-Team: ja <LL@li.org>
Plural-Forms: nplurals=1; plural=0;
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.16.0
 **dict[str, Any]** -- ãã©ã¡ã¼ã¿ :class:`Evaluator` ã®ãã©ã¡ã¼ã¿ãç¢ºèªããã ã³ãã³ãã©ã¤ã³å¼æ°ã¯ãã¡ã¤ã«ãã©ã¡ã¼ã¿ã§ä¸æ¸ãããããã¤ã¾ãããã¡ã¤ã«ãã©ã¡ã¼ã¿ã§ãã¹ã¦è¨­å®ããå ´åãå¿ãããã³ãã³ãã©ã¤ã³å¼æ°ãä½¿ç¨ããå¿è¦ã¯ãªãã å­¦ç¿æ¸ã¿ã¢ãã«ã®è©ä¾¡ãã«ã¹ã¿ãã¤ãºããã ã¢ãã«ã®è©ä¾¡ãè¡ãã è©ä¾¡ãã¼ã¿ãèª­ã¿è¾¼ãã ä»ã«å¿è¦ãªãã©ã¡ã¼ã¿ã¯ããã¡ã¤ã«ãã©ã¡ã¼ã¿ã§è¨­å®ããã ãã©ã¡ã¼ã¿ æ»ãå¤ ``.compile()`` ãå®è¡ããã ``.summary()`` ãå®è¡ããã è©ä¾¡ãå®è¡ããã å±éãªãã©ã¡ã¼ã¿ã®ã¿ãã³ãã³ãã©ã¤ã³å¼æ°ã§è¨­å®ããã ã¢ãã«ã®éã¿ãè¨­å®ããã ã³ãã³ãã©ã¤ã³å¼æ°ã¨ãã¡ã¤ã«ãã©ã¡ã¼ã¿ãè¨­å®ããã èª¤å·®é¢æ°ãã¢ãã«ãã¡ããªã¯ã¹ãè¨­å®ããã è©ä¾¡ã®è¨­å®ãè¡ãã ``@save_params_log`` ã¨ ``@process_time`` ã§ãã³ã¬ã¼ãããã¦ããã ã¢ãã«ãè©ä¾¡ããã¢ã¸ã¥ã¼ã«ã ãã¡ã¤ã«ã¯ãyamlãã¡ã¤ã«ãä½¿ç¨ããã(:func:`lib.common.file.load_yaml`) ã¯ã©ã¹ã®ä¸è¦§ key=loss: èª¤å·®é¢æ°ã¯ã©ã¹ key=metrics: ã¡ããªã¯ã¹ã¯ã©ã¹ã®ãªã¹ã key=opt: æé©åææ³ã¯ã©ã¹ ã³ã¼ã«ããã¯ã¯ã©ã¹ã®ãªã¹ã ã¢ãã«ã¯ã©ã¹ ãã©ã¡ã¼ã¿ å 