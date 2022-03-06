
// Import all the packages and header files.
// Import the header file "LBFGS.h" that contains the LBFGS method.
// Import the "Curves.h" file to create a Piecewise Curve of volatilities.

#include <iostream>
#include <aadc/ibool.h>
#include <aadc/aadc.h>
#include <random>
#include <aadc/aadc_matrix.h>
#include <thread>
#include "examples.h"
#include "Curves.h"
#include <LBFGS.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using Eigen::VectorXd;

using namespace aadc;

using namespace LBFGSpp;

typedef double Time;

typedef __m256d mmType;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Function_2P { 

public:

    // Create a stucture to store all the variables for each European Option (strike price, interest rate,
    // asset price, observed price and expected price).
    template<class vtype>
    struct EurOption{

        EurOption(const double maturity_, const vtype strike_, const double vol_, const double rate_
                        , const double asset_, const int paths_) : maturity(maturity_), strike(strike_), init_vol(vol_)
                        , init_rate(rate_), init_asset(asset_), paths(paths_) {}

        int paths;

        double maturity, grad, init_vol, init_rate, init_asset, obs_price, total_price
            , variance;

        ScalarVector payoffs = ScalarVector(paths, 0.);
        
        vtype strike;

    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //Initialize of an empty vector of European Options.
    std::vector<EurOption<idouble>> eos;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // This function represents the objective funtion G = (1/2)* sum_{i=0}^{number_of_options} (Ey_i - C_i).
    inline double loss() {

        double sum(0.);

        for (int it_options = 0; it_options < num_options; it_options++) {
            
            sum += std::pow(eos[it_options].total_price - eos[it_options].obs_price, 2);

        }

        return sum/2;

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Creates a contanst interest rate function.
    template<class vtype>
    class BankRate{
        
        public: 
            
            BankRate (vtype _rate) : rate(_rate) {}

            vtype operator () (const Time& t) const {return rate;}

            ~BankRate() {}
        
        public:
            
            vtype rate;

            std::vector<double> time_vec;

            std::vector<vtype> rate_vals;

    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Calculates the asset price for a certain time and a random Monte Carlo path.
    template<class vtype, class VolClass>
    vtype simulateAssetOneStep(
        const vtype current_value
        , Time current_t
        , Time next_t
        , const BankRate<vtype>& rate
        , const VolClass& vol_obj
        , const vtype& random_sample
    ) {
        
        double dt = (next_t-current_t);

        // Interpolation knot of the Piecewise Linear Curve of volatilities.
        vtype vol = vol_obj(current_t);

        // Formula calculating the asset price at a certain time.
        vtype next_value = current_value * (
            std::exp((-vol*vol / 2 + rate(current_t))*dt  +  vol * std::sqrt(dt) * random_sample)
        );

        return next_value;

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Calculates the payoffs (S(t_i) - K)^+ for all times t_i.  
    template<class vtype, class VolClass>
    void onePathPricing(
        vtype asset
        , const BankRate<vtype>& rate_obj
        , const VolClass& vol_obj
        , const iVector& random_samples
        , iVector& payoffs
    ) {

        int it_options = 0, t_i = 0;

        while (it_options < num_options) {

            t_i = 0;
        
            while (t_i < time_list.size()-1) {
    
                // Call options formula (S(t_i) - K)^+ = max(S(t_i) - K, 0).
                asset = simulateAssetOneStep(
                    asset, time_list[t_i], time_list[t_i+1], rate_obj, vol_obj, random_samples[t_i]
                );

                if (eos[it_options].maturity == time_list[t_i + 1]) {
                
                    payoffs[it_options] = std::max(asset - eos[it_options].strike, 0.);

                    t_i = time_list.size();

                    it_options++;

                } else t_i++;

            }
            
        }

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Function_2P () {
            
        idouble aad_rate(init_rate), aad_asset(init_asset);

        iVector strike_list = {95., 105., 115., 90., 100.}, aad_vol, aad_random_samples;

        for (int it_options = 0; it_options < num_options; it_options++) {
        
            eos.push_back(
                EurOption<idouble>(
                    time_list2[it_options+1], strike_list[it_options]
                    , init_vol_list[it_options], init_rate, init_asset
                    , paths_per_thread
                )
            );

        }

        aad_vol.resize(num_options), aad_random_samples.resize(num_options+1);
        
        aad_funcs.startRecording();
            // Mark the vector of random variables as input only, no adjoints for them.
            markVectorAsInput(random_arg, aad_random_samples, false); 

            // Mark the vector of random variables as input, adjoints for them.
            markVectorAsInput(vol_arg, aad_vol, true);
            
            // Marks the value as input but the derivative is not required.
            rate_arg = aad_rate.markAsInput();

           // Marks the value as input but the derivative is not required.
            asset_arg = aad_asset.markAsInput();
            
            BankRate<idouble> rate_obj(aad_rate);

            // Creates a Piecewise Linear curve of volatilities.
            PiecewiseLinearCurve<idouble> vol_obj(time_list2, aad_vol);

            iVector payoffs(num_options);

            // Calls the funtion to calculate the payoffs for one Monte Carlo path
            onePathPricing(aad_asset, rate_obj, vol_obj, aad_random_samples, payoffs);

            // All the variables subject to differentiation.
            markVectorAsOutput(payoffs_args, payoffs);         

        aad_funcs.stopRecording();

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
    // Prints the results and the corresponding variables for all the European Options.
    void Print_Results(const VectorXd& x, bool result, const int niter) {

        for (int it = 0; it < num_options; it++){
            
            std::cout << "Num " << it+1 <<"\n";
            std::cout << "-------------------\n";
            std::cout << "Strike= " << eos[it].strike << "\n";
            std::cout << "Matur = " << eos[it].maturity << "\n";
                
            // It prints the obseved prices if the "result" is true or prints the expectated prices if
            // the "result" is false.
            if (result == true) { std::cout << "Obser Price = " << eos[it].obs_price << "\n"; }
            else { std::cout << "Obser Price = " << eos[it].total_price << "\n"; }
            std::cout << "Vol = " << x[it] << "\n";

        }
        
        // Prints the result of "G" in loss().
        if (result == false) { std::cout << "\nLoss " << niter << " = " << loss() << "\n\n"; }
        else { std::cout << "\n"; }

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Prints the results and the corresponding variables for all the European Options.
    void Print_Gradient() {

        //Prints the gradient of "G", which is represented by loss().
        std::cout << "Gradient = ( ";
        for (int it_options = 0; it_options < num_options-1; it_options++) {
            
            std::cout << eos[it_options].grad << ", ";

        }

        std::cout << eos[num_options-1].grad << ")\n\n";

        // Prints the standard deviation and variance for each entry of the gradient.
        for (int it_options = 0; it_options < num_options; it_options++) {

            std::cout << "Standard Deviation of the "<< it_options+1 <<" entry of the gradient = " <<
                sqrt(eos[it_options].variance/num_mc_paths) << "\n";

            std::cout << "Variance of the "<< it_options+1 <<" entry of the gradient = " <<
                eos[it_options].variance/num_mc_paths << "\n\n";

        }

        /*std::cout << "***************************************" << std::endl;
        std::cout << "Code size forward : " << aad_funcs.getCodeSizeFwd() << std::endl;
        std::cout << "Code size reverse : " << aad_funcs.getCodeSizeRev() << std::endl;
        std::cout << "Work array size   : " << aad_funcs.getWorkArraySize() << std::endl;
        std::cout << "Stack size        : " << aad_funcs.getStackSize() << std::endl;
        std::cout << "Const data size   : " << aad_funcs.getConstDataSize() << std::endl;
        std::cout << "Check point size  : " << aad_funcs.getNumCheckPoints() << std::endl;
        std::cout << "Check memory use  : " << aad_funcs.getMemUse() << std::endl;
        std::cout << "Check workspace memory use  : " << aad_funcs.getWorkSpaceMemUse() << std::endl;
        std::cout << " Code size forward : " << aad_funcs.getCodeSizeFwd() << std::endl;
        std::cout << " Code size reverse : " << aad_funcs.getCodeSizeRev() << std::endl;
        std::cout << " Work array size   : " << aad_funcs.getWorkArraySize() << std::endl;
        std::cout << " Stack size        : " << aad_funcs.getStackSize() << std::endl;
        std::cout << " Const data size   : " << aad_funcs.getConstDataSize() << std::endl;
        std::cout << " CheckPoint size   : " << aad_funcs.getNumCheckPointVars() << std::endl;
        std::cout << " Func + WS mem use : " << aad_funcs.getMemUse()+aad_funcs.getWorkSpaceMemUse() << " bytes" << std::endl;
        std::cout << "*************************************" << std::endl;*/

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // This is an auxilary function for calcuting the expectation or the gradient, depending on the value of "step".
    // It performs the Monte Carlo simulation using parallezation for a faster performance.

    void ThreadFuction(
        const VectorXd& vol_vector
        , const bool step
        , ScalarMatrix& grad_vector
        , ScalarVector& price
        , ScalarVector& grad_total
    ) {
        
        // Initialization of certain variables and the Normal Distribution N(0, 1).
        ScalarVector prices(num_options, 0.)
                    , vol_vec(num_options, 0.);

        for (int it = 0; it < num_options; it++) vol_vec[it] = vol_vector[it];

        std::uniform_real_distribution<> normal_distrib(0.0, 1.0); 
            
        AVXVector<mmType> mm_price(num_options,mmSetConst<mmType>(0.))
                        , mm_grad(num_options,mmSetConst<mmType>(0.))
                        , randoms(num_options+1);

        // This instruction creates the execution context, this is just a tape of mmType variables.    
        ws = aad_funcs.createWorkSpace(); 

        gen.seed(4*17+31);

        for (int mc_i = 0; mc_i < paths_per_thread; ++mc_i) {

            for (int it_options = 0; it_options < num_options+1; it_options++) {

                for (int c = 0; c < AVXsize; c++) toDblPtr(randoms[it_options])[c] = normal_distrib(gen);

            }

            // All variables marked as Input have to be initialized before calling
            // the forward algorithm.         
            setAVXVector(*ws, random_arg, randoms);
            setScalarVectorToAVX(*ws, vol_arg, vol_vec);
            ws->setVal(rate_arg, mmSetConst<mmType>(init_rate));
            ws->setVal(asset_arg, mmSetConst<mmType>(init_asset));

            // This instruction calls the Forward algorithm with a preallocated execution context ws.
            (aad_funcs).forward(*ws);

            // Resets all the previouslly created adjoints.
            ws->resetDiff();

            // if "step" == true, then it will only calculate the observed prices.
            if (step == true) {
                
                for (int it_options = 0; it_options < num_options; it_options++) {

                    // ws->val(...) will contain the final value of G.    
                    mm_price[it_options] = aadc::mmAdd(
                        mm_price[it_options], ws->val(payoffs_args[it_options])
                    );

                }

            } else {

                for (int it_options = 0; it_options < num_options; it_options++){
                    
                    // Since we are implementing adjoint differentiation, we need to to initialize the adjoint
                    // variables (according to the second algorothm).
                    mm_price[it_options] = aadc::mmAdd(
                        mm_price[it_options], ws->val(payoffs_args[it_options])
                    );

                    eos[it_options].payoffs[mc_i] = 0.;

                    eos[it_options].payoffs[mc_i] = aadc::mmSum(
                        ws->val(payoffs_args[it_options])
                    )/(AVXsize);

                }

                if (mc_i != 0){  

                    for (int it_options = 0; it_options < num_options; it_options++){

                        ws->setDiff(
                            payoffs_args[it_options], eos[it_options].payoffs[mc_i-1] - eos[it_options].obs_price
                        );

                    }

                    // This calls the reverse AD algorithm.           
                    (aad_funcs).reverse(*ws);

                    for (int it_options = 0; it_options < num_options; it_options++){
                        
                        // ws->diff(...) will contain the derivative of G_sigma.
                        mm_grad[it_options] = aadc::mmAdd(
                            ws->diff(vol_arg[it_options]), mm_grad[it_options]
                        );

                        grad_vector[it_options][mc_i-1] = aadc::mmSum(
                            ws->diff(vol_arg[it_options])
                        )/(AVXsize);

                    }
                        
                }
                    
            }

        }

        if (step == true) {
            
            // Sums up all the mmType variables.    
            for (int it = 0; it < num_options; it++) price[it] = aadc::mmSum(mm_price[it])/(num_mc_paths);

        } else {

            for (int it_options = 0; it_options < num_options; it_options++){
                    
                price[it_options] = aadc::mmSum(mm_price[it_options])/(num_mc_paths);
                    
                grad_total[it_options] = aadc::mmSum(mm_grad[it_options])/(num_mc_paths-1);
                    
            }

        }

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // This function is used to calculate the observed prices since in our case the European Options are fictional.
    void operator()(const VectorXd& vol_list_obs) {

        ScalarVector price_obs1(num_options, 0.)
            , grad_total1(num_options, 0.);

        ScalarMatrix grad_vector1(num_options, paths_per_thread-1, 0.);

        // With "true", calculate only the observed prices.
        ThreadFuction(vol_list_obs, true, grad_vector1, price_obs1, grad_total1);
        
        // Store the results for the observed prices for each European Option.
        for (int it = 0; it < num_options; it++) eos[it].obs_price = price_obs1[it];

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    double operator()(const VectorXd& x, VectorXd& grad)
    {
        ScalarVector exp_price(num_options, 0.)
            , grad_total(num_options, 0.)
            , stand_deriv(num_options, 0.);

        ScalarMatrix grad_vector(num_options, paths_per_thread-1, 0.);

        // With "false", calculate only the gradient of G.
        ThreadFuction(x, false, grad_vector, exp_price, grad_total);

        // Store the results and calculate the variance of the gradients.
        for (int it_options = 0; it_options < num_options; it_options++) {

            eos[it_options].total_price = exp_price[it_options];

            grad[it_options] = 0.; eos[it_options].grad = 0.;

            eos[it_options].grad = grad_total[it_options];

            grad[it_options] = grad_total[it_options];

            eos[it_options].variance = 0.;

            // Calculates the variance of each entry of the gradient.
            for (int mc = 0; mc < grad_vector.cols(); mc++) {

                eos[it_options].variance += std::pow(
                    grad_vector[it_options][mc] - grad[it_options], 2
                );
                        
            }

            eos[it_options].variance /= num_mc_paths;

        }

        double _f = loss();

        return _f;

    }

private:

    // This section initialises all variables for the set of European Options: 
    // - the number of options (5 European options)
    // - the interest rate (r) (0.1 for all Options)
    // - the initial asset price (S_0) (100 for all Options)
    // - strike prices (K) (95, 105, 115, 90 and 100)
    // - maturity times (T) (1, 2, 3, 4 and 5)
    // - a set of initial volatilities (vol) to calculate some artifical
    // observed prices (C) (0.005, 0.004, 0.0045, 0.003 and 0.0035) 
    // - 40000000 Monte Carlo paths.

    aadc::AADCFunctions<mmType> aad_funcs;

    std::shared_ptr<aadc::AADCWorkSpace<mmType> > ws;

    aadc::VectorArg random_arg, vol_arg;

    aadc::AADCArgument rate_arg, asset_arg;

    aadc::VectorRes payoffs_args;

    int num_options = 5, num_mc_paths = 40000000, AVXsize = aadc::mmSize<mmType>()
        , iterations, paths_per_thread = num_mc_paths / AVXsize;

    double init_rate = 0.1, init_asset = 100.;

    ScalarVector init_vol_list = {0.005, 0.004, 0.0045, 0.003, 0.0035}
                , time_list2 = {0, 1, 2, 3, 4, 5};

    std::vector<Time> time_list = {0, 1, 2, 3, 4, 5};

    std::mt19937_64 gen;

};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void option_pricing_2_Proj()
{   

    // Initializes the number of iterations. 
    int iterations_2P;
    std::cout << "Number of iterations: ";
    std::cin >> iterations_2P;

    // Sets up parameters.
    LBFGSParam<double> param_2P;
    param_2P.epsilon = 1e-9;
    param_2P.max_iterations = iterations_2P;
    
    // Creates the solver and function object.
    LBFGSSolver<double> solver_2P(param_2P);

    // Volatilities for the observed prices.
    ScalarVector init_vol_list = {0.005, 0.004, 0.0045, 0.003, 0.0035};
    VectorXd x = VectorXd::Zero(5);
    VectorXd grad = VectorXd::Zero(5);
    for (int it = 0; it < init_vol_list.size(); it++) x[it] = init_vol_list[it];
    
    Function_2P fun;

    double fxx;

    // Calculates and prints the observed prices.
    fun(x);
    fun.Print_Results(x, true, 0);

    // Initial guess.
    for (int it = 0; it < x.size(); it++) x[it] *= (1+0.2*(double(std::rand()) / RAND_MAX));

    // Calculates and prints the expected prices and the gradient of G.
    fxx = fun(x, grad);
    fun.Print_Results(x, false, 0);
    fun.Print_Gradient();

    double fx;

    auto base_start = std::chrono::high_resolution_clock::now();

    // x will be overwritten to be the best point found.
    int niter = solver_2P.minimize(fun, x, fx);
    fun.Print_Results(x, false, niter);
    fun.Print_Gradient();
    
    auto base_stop = std::chrono::high_resolution_clock::now();
    
    std::chrono::microseconds base_time = std::chrono::duration_cast<std::chrono::microseconds>(base_stop - base_start);
    
    std::cout << "Base time (double): " << niter << " iterations = " << base_time.count() << " microseconds\n\n";

}