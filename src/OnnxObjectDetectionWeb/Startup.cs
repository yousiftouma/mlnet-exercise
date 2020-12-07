using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.ML;
using OnnxObjectDetectionWeb.Services;
using OnnxObjectDetection;
using OnnxObjectDetection.ML;
using OnnxObjectDetection.ML.DataModels;
using OnnxObjectDetectionWeb.ImageFileHelpers;
using OnnxObjectDetectionWeb.Utilitites;

namespace OnnxObjectDetectionWeb
{
    public class Startup
    {
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;

            //TODO create an ML.NET pipeline model and save the model as a zip 
        }

        public IConfiguration Configuration { get; }

        // This method gets called by the runtime. Use this method to add services to the container.
        public void ConfigureServices(IServiceCollection services)
        {
            services.Configure<CookiePolicyOptions>(options =>
            {
                // This lambda determines whether user consent for non-essential cookies is needed for a given request.
                options.CheckConsentNeeded = context => true;
                options.MinimumSameSitePolicy = SameSiteMode.None;
            });

            services.AddControllers();
            services.AddRazorPages();

            //TODO load your model from the zip you saved and register a PredictionEnginePool in DI

            services.AddTransient<IImageFileWriter, ImageFileWriter>();
            services.AddTransient<IObjectDetectionService, ObjectDetectionService>();
        }

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            if (env.EnvironmentName == Environments.Development)
            {
                app.UseDeveloperExceptionPage();
            }
            else
            {
                app.UseExceptionHandler("/Error");
            }

            app.UseStaticFiles();
            app.UseCookiePolicy();

            app.UseRouting();
            app.UseEndpoints(endpoints => {
                endpoints.MapControllers();
                endpoints.MapRazorPages();
            });
        }       
    }
}
