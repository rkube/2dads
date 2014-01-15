/*
 * Array used in diagnostic functions
 *
 * From array_base, but includes function to compute mean, fluctuatioons and stuff
 * diag_array is derived from a templated class. Thus all members of the base class
 * are unknown to the compiler.
 * Paragraph 14.6/3 of the C++11 Standard:
 */

#ifndef DIAG_ARRAY_H
#define DIAG_ARRAY_H

#include "error.h"
#include "check_bounds.h"
#include "2dads_types.h"
#include "array_base.h"
#include "cuda_array2.h"
#include <cstring>

using namespace std;


template <class T>
class diag_array : public array_base<T>
{
    public:
        diag_array(cuda_array<T>&);

        // Declare as virtual since they can be updated by child classes
        diag_array<T>& operator=(const T&);
        diag_array<T>& operator=(const diag_array<T>&);
        // Copy functions

        // Inlining operator<<, see http://stackoverflow.com/questions/4660123/overloading-friend-operator-for-template-class/4661372#4661372
        friend std::ostream& operator<<(std::ostream& os, const diag_array<T>& src)                             
        {                                                                                                    
            //const unsigned int tl{src.get_tlevs()};                                                          
            //const unsigned int nx{src.get_nx()};                                                             
            //const unsigned int my{src.get_my()};                                                             
            uint tl = src.get_tlevs();
            uint nx = src.get_nx();
            uint my = src.get_my();

            for(unsigned int t = 0; t < tl; t++)                                                             
            {                                                                                                
                os << "t: " << t << "\n";                                                                    
                for(unsigned int n = 0; n < nx; n++)                                                         
                {                                                                                            
                    for(unsigned int m = 0; m < my; m++)                                                     
                    {                                                                                        
                    os << src(t,n,m) << "\t";                                                            
                }                                                                                        
                os << "\n";                                                                              
            }                                                                                            
            os << "\n\n";                                                                                
            }                                                                                                
            return (os);                                                                                     
        }                                          

        void update(cuda_array<T>&);
};

#endif //DIAG_ARRAY_H
