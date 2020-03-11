# Everything You've Wanted to Know About Scope*
### *But were too afraid to ask 

### Python Scope Basics

Scope is about the meaning of variables in your code. Python's scopes are the places where variables are defined and looked up. Use scope to your advantage to prevent name clashes in your program, minimize program maitenence efforts, and retain state information between function calls.

When you use a name in a program, Python creates, changes, or looks up the name in a "namespace". A namespace is the location of a name's assignment in your source code.  Names must be assigned before they can be used, and Python uses the assignment location of a name to determine the scope in which it can be used.



Variables can be defined in 3 different places:

- Local variables are assigned within a def
- Nonlocal variables are assigned within a nested def 
- Global variables are assigned outside all defs


##### "The place where you assign a name in your code determines the namespace it will live in, and hence its scope of visibility," (Lutz, 506)


```python
X = 99 #<----- Global (module) scope 

def func1():
    X = 88 #<---- Local (function) scope. This is a different variable.
    
print(X)
```

    99



```python
L = [1,2,3]

def func2():
    L.append(X) #<--- in place changes do not qualify variables as Local. Changing an object != to assigning a variable
    
    #L = 123  
func()
L
```




    [1, 2, 3, 99]




```python
def pandas_test():
    import pandas as pd
    return pd.DataFrame(L) #<--- pd is good to go here. 


print(pandas_test())       #<---- scope is determined at assignment time, 
                           #      but a function call in the global scope references the appropiate local scope. 


pd.DataFrame(L)            #<--- global scope doesn't know what pd is. 
```

        0
    0   1
    1   2
    2   3
    3  99



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-67-2df16c9f43c5> in <module>
          8 
          9 
    ---> 10 pd.DataFrame(L)            #<--- global scope doesn't know what pd is.
    

    NameError: name 'pd' is not defined


Local variables are only needed while the function is running They're "removed from memory when the function call exits, and the objects they reference may be garbage-collected if not referenced elsewhere," (Lutz 511).

### Name Resolution: The LEGB Rule

When you reference a variable, Python will search for that name in up to four namespaces:

- Local 
- Enclosing functions
- Global 
- Built-in

(this rule has exceptions including attribute names, comprehension variables, and exception variables)

### The Built-in Scope

- The built-in scope is a built-in module called builtins. But to see it, you have to import 'builtins'
- Because these names are in the built-in scope, you can use them without importing any modules.
    - i.e., you don't have to say builtin.dict() 
- Since B is the last step in the LEGB look-up process, it's possible to override these names
    - but should you? 


```python
import builtins
dir(builtins)
```




    ['ArithmeticError',
     'AssertionError',
     'AttributeError',
     'BaseException',
     'BlockingIOError',
     'BrokenPipeError',
     'BufferError',
     'BytesWarning',
     'ChildProcessError',
     'ConnectionAbortedError',
     'ConnectionError',
     'ConnectionRefusedError',
     'ConnectionResetError',
     'DeprecationWarning',
     'EOFError',
     'Ellipsis',
     'EnvironmentError',
     'Exception',
     'False',
     'FileExistsError',
     'FileNotFoundError',
     'FloatingPointError',
     'FutureWarning',
     'GeneratorExit',
     'IOError',
     'ImportError',
     'ImportWarning',
     'IndentationError',
     'IndexError',
     'InterruptedError',
     'IsADirectoryError',
     'KeyError',
     'KeyboardInterrupt',
     'LookupError',
     'MemoryError',
     'ModuleNotFoundError',
     'NameError',
     'None',
     'NotADirectoryError',
     'NotImplemented',
     'NotImplementedError',
     'OSError',
     'OverflowError',
     'PendingDeprecationWarning',
     'PermissionError',
     'ProcessLookupError',
     'RecursionError',
     'ReferenceError',
     'ResourceWarning',
     'RuntimeError',
     'RuntimeWarning',
     'StopAsyncIteration',
     'StopIteration',
     'SyntaxError',
     'SyntaxWarning',
     'SystemError',
     'SystemExit',
     'TabError',
     'TimeoutError',
     'True',
     'TypeError',
     'UnboundLocalError',
     'UnicodeDecodeError',
     'UnicodeEncodeError',
     'UnicodeError',
     'UnicodeTranslateError',
     'UnicodeWarning',
     'UserWarning',
     'ValueError',
     'Warning',
     'ZeroDivisionError',
     '__IPYTHON__',
     '__build_class__',
     '__debug__',
     '__doc__',
     '__import__',
     '__loader__',
     '__name__',
     '__package__',
     '__spec__',
     'abs',
     'all',
     'any',
     'ascii',
     'bin',
     'bool',
     'breakpoint',
     'bytearray',
     'bytes',
     'callable',
     'chr',
     'classmethod',
     'compile',
     'complex',
     'copyright',
     'credits',
     'delattr',
     'dict',
     'dir',
     'display',
     'divmod',
     'enumerate',
     'eval',
     'exec',
     'filter',
     'float',
     'format',
     'frozenset',
     'get_ipython',
     'getattr',
     'globals',
     'hasattr',
     'hash',
     'help',
     'hex',
     'id',
     'input',
     'int',
     'isinstance',
     'issubclass',
     'iter',
     'len',
     'license',
     'list',
     'locals',
     'map',
     'max',
     'memoryview',
     'min',
     'next',
     'object',
     'oct',
     'open',
     'ord',
     'pow',
     'print',
     'property',
     'range',
     'repr',
     'reversed',
     'round',
     'set',
     'setattr',
     'slice',
     'sorted',
     'staticmethod',
     'str',
     'sum',
     'super',
     'tuple',
     'type',
     'vars',
     'zip']




```python
open = 99 
open
```




    99



### The Global Statement 

- "The global statement tells Python that a function plans to change one or more global names," (Lutz, 515)
- Global and it's cousin, non-local, are examples or namespace declarations
- Global names must be declard as such only if they're assigned within a function
    - By contrast, global names may be refrerenced in a function without a declaration


```python
X = 88 

def func4():
    global X 
    X = 99 
    
func4()
print(X)
```

    99


#### Program Design: Minimize Global Variables 

"In general, functions should rely on arguments and return values instead of gloabls," (Lutz, 516)

- The extra work to declare a global variable in a local scope is deliberate. 
    - you have to say more to do the potentially 'wrong' thing
- Changing globals can lead to well-known design problems
    - The value that a variable represents is now dependent on the order of function calls
    - Things will get confusing quickly, especially for someone trying to understand your code later on

    


```python
X = 99

def func5():
    global X 
    X = 88 
    
def func6(): 
    global X 
    X = 77

# What will be the value of X here? To answer that question, you have trace the flow of the entire code. 
```

#### Program Design: Minimize Cross-File Changes 

- We can change variables from another file, but we normally shouldn't
- Seperating variables on a per-file basis is useful to avoid name-clashes
    - The global of a module file becoes the attribute namespace of the module object once it's imported 
    - The imported file's global scope is transformed into the resulting object's attribute namespace 
- After importing a file, it is possible to change that file's namespace
    - It's generally best to avoid this in favor of functions that pass-in arguments and return values
    


```python
## Not recomended. 

# first.py 
X = 99 

# second.py 
import first 
print(first.X)
first.X = 88 
```


```python
## "Accessor Function": A better way 

# first.py 
X = 99 

def setX(new):    #<----Accessor functions make your intentions for an external change implicit 
    global X 
    X = new

# second.py 
import first 
first.setX(88)    #<----More readable and maintainable than changing the variable directly
```

### Nested Scope

- Nested Scope refers to the 'E' in LEGB: Enclosing functions
- Enclosing scopes take the form of local scope + all enclosing functions' scope's
    - sometimes called 'statically nested scopes
- A reference in a nested scope will look first in the local scope, then in the enclosing scope, and so on
- An assignment in a nested scope can be declard global or nonlocal 
    - nonlocal will change the variable in the closest enclosing scope 
- lambda functions often employ nested scope


```python
X = 99 

def f1():
    X = 88 
    def f2():
        print(X)   #<---- f2 finds X in enclosing function
    return f2      #<---- return f2, but don't call it 

action = f1()
action()           #<---- the call to action is really running the function we named f2 when f1 ran
```

    88


### Factory Functions: Closures

- The code above is sometimes described as a closure or a factory function
    - i.e. a function that makes a function 
- A factory function object 'remembers' values in enclosing functions 
    - this is called 'state-retention, and it's a common theme
- State-retained values are local to each copy of the nested function created
- Factory functions can provide a simple alternative to classes
- They're used when programs that need to generate an object on the fly in response to conditions at runtime 
    - e.g. a GUI that must define actions according to user inputs that can't be anticipated 
    - in this case, we need a function that makes a function


```python
def maker(N):
    def action(X):
        return X ** N
    return action 
```


```python
f = maker(2) # N = 2 in f's scope  
f # we get back a reference to the generated nested function
```




    <function __main__.maker.<locals>.action(X)>




```python
f(3) # pass 3 to X, 'action' retains the scope of it's enclosing function (N = 2)
```




    9



### Closures vs. Classes 

- Classes are better at state retention than factory functions because they:<br>  
    - make their memory more explicit with attribute assignments 
    - directly support additional tools like customization by inheritence and operator overloading <br>
        - inheritance = classes on classes 
        - operator overloading e.g., + can mean add or concat <br>
- Factory functions provide a light-weight, viable alternative when retaining state is the only goal

#### In conclusion, 

- Normal local variables go away after the function is called

- Values can be retained from call to call by:
   
    1) assigning variables in the global scope 
    
    2) explicitly using 'global' or 'nonlocal'
    
    3) enclosing scope references (closures)
    
    3) using class attributes 
    
    4) argument defaults and function attributes (not-pictured here) 


```python

```
