#ifndef MFIX_LEVEL_H_
#define MFIX_LEVEL_H_

#include <iostream>
#include <memory>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <AMReX_AmrCore.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_PhysBCFunct.H>

#include <AMReX_EBFabFactory.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF_Plane.H>
#include <AMReX_EB2_IF_Polynomial.H>
#include <AMReX_EB2_IF_Translation.H>
#include <AMReX_EB2_IF_Intersection.H>
#include <AMReX_EB_levelset.H>
#include <AMReX_EB_LSCore.H>

#include <MFIXParticleContainer.H>
#include <mfix_des_F.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLaplacian.H>
#include <mfix_eb_if.H>
#include <MFIX_BcList.H>

enum DragType
{
  Invalid=-1,
  WenYu,
  Gidaspow,
  BVK2,
  UserDrag
};

class mfix
    : public AmrCore
{
public:

    mfix ();
    ~mfix ();

    // Face-based coefficients b in MAC projection and implicit diffusion solve
    Vector< Array< std::unique_ptr<MultiFab>, AMREX_SPACEDIM> > bcoeff_cc;

    // LinOpBCType Definitions are in amrex/Src/Boundary/AMReX_LO_BCTYPES.H
    std::array<amrex::LinOpBCType,AMREX_SPACEDIM>  ppe_lobc;
    std::array<amrex::LinOpBCType,AMREX_SPACEDIM>  ppe_hibc;

    void InitParams(int solve_fluid, int solve_dem, int call_udf );

    bool IsSteadyState() { return (steady_state > 0); }

    void Init(Real time);

    void InitLevelData(Real time);

    void mfix_setup_nodal_solver();

    void PostInit(Real& dt, Real time, int restart_flag, Real stop_time);

    void ResizeArrays();

    void WriteCheckPointFile(std::string & check_file_name,
                             int nstep = 0, Real dt = 0.0, Real time = 0.0)  const;

    void WritePlotFile(std::string & plot_file_name,
                       int nstep = 0, Real dt = 0.0, Real time = 0.0) const;

    //! Save variables that don't change to plot file. The idea is that they can
    //! be saved _once_ per simulations and not repeatedly every Nth time step.
    void WriteStaticPlotFile(const std::string & plot_file_name) const;

    void Restart(std::string & restart_chkfile, int * nstep, Real * dt, Real * time,
                 IntVect & Nrep);

    void WriteParticleAscii(std::string & par_ascii_file_name, int nstep = 0) const;

    void WriteAverageRegions ( std::string& avg_file, int nstep, Real time = 0.0 ) const;

    void ComputeAverageFluidVars ( const int lev,
                                   const amrex::Real time,
                                   const std::string&  basename,
                                   const Vector<int>& avg_p_g,
                                   const Vector<int>& avg_ep_g,
                                   const Vector<int>& avg_vel_g,
                                   const Vector<Real>& avg_region_x_w,
                                   const Vector<Real>& avg_region_x_e,
                                   const Vector<Real>& avg_region_y_s,
                                   const Vector<Real>& avg_region_y_n,
                                   const Vector<Real>& avg_region_z_b,
                                   const Vector<Real>& avg_region_z_t ) const;

    void Regrid ();

    void Evolve(int nstep, Real & dt, Real & prev_dt, Real time, Real stop_time);

    void mfix_usr1_cpp(amrex::Real* time);

    void usr3();

    void output(int estatus, int finish, int nstep, Real dt, Real time)
    {
        pc->output( estatus, finish, nstep, dt, time);
    };

    void make_mg_bc(int mg_bc[]);

    void make_eb_geometry ();
    void make_eb_factories ();
    void fill_eb_levelsets ();
    void intersect_ls_walls ();

    template<class F> void build_particle_eb_levels (EB2::GeometryShop<F> gshop)
    {
        if(solve_dem)
        {
            for (int lev = 0; lev < nlev; lev ++)
                particle_eb_levels[lev] = LSCore<F>::BuildEBLevel(gshop,
                                                                  lev, geom, 100, 0);
        }
    }


    //! Construct EB levels from Geometry shop. This builds each EB level and
    //! saves a pointer to each level into `eb_levels`.
    template<class F> void build_eb_levels (EB2::GeometryShop<F> gshop)
    {

        /************************************************************************
         *                                                                      *
         * Build EB levels                                                      *
         *                                                                      *
         ***********************************************************************/

        for (int lev = 0; lev < nlev; lev ++)
        {
            // NOTE: the last two parameters are the max coarsening level and
            // the required coarsening level. They are also used by the Poisson
            // solver, => changing these will result in slight changes of the
            // fluid solve.
            eb_levels[lev] = LSCore<F>::BuildEBLevel(gshop, lev, geom, 100, 0);
            particle_eb_levels[lev] = eb_levels[lev];
        }

        if (nlev == 1)
        {
            Box dom_ls = geom[0].Domain();
            dom_ls.refine(levelset__refinement);
            Geometry geom_ls(dom_ls);
            Vector<Geometry> vgeom{geom_ls};
            // NOTE: lev here is index in vgeom
            eb_levels[1] = LSCore<F>::BuildEBLevel(gshop, 0, vgeom, 100, 0);
            particle_eb_levels[1] = eb_levels[1];
        }
    }

    void get_input_bcs();

    void set_input_bcs(const std::string bcID, const int index,
                       const int cyclic, const Real domloc);

    std::unique_ptr<UnionListIF<EB2::PlaneIF>> get_walls(bool & has_walls);

    std::unique_ptr<UnionListIF<EB2::PlaneIF>> get_real_walls(bool & has_real_walls);

    std::unique_ptr<
        EB2::TranslationIF<EB2::PolynomialIF>
        > get_poly(int max_order, std::string field_prefix);

    std::unique_ptr<EB2::IntersectionIF<EB2::PlaneIF,EB2::PlaneIF,EB2::PlaneIF>>
        make_wall(int dir, // direction (long edge) of wall
                  Real position, Real height, Real width );

    void WriteMyEBSurface();

    void InitIOChkData();
    void InitIOPltData();

    static std::string get_load_balance_type();

    // flag enabling level-set restart (i.e. prevent make_eb_* from rebuilding
    // the level-set data).
    bool levelset__restart = false;

    void mfix_compute_vort();

    void mfix_compute_dt(int nstep, Real time, Real stop_time, Real & dt);

    ///
    /// Parameters describing the level-set grid:
    ///
    int levelset__refinement = 1;    // refinement (wrt particle grid) of the level-set's grid
    int levelset__eb_refinement = 1; // refinement of the EB facets used to generate the level-set
    int levelset__pad = 2;           // padding of the level-set grid
    int levelset__eb_pad = 2;        // padding of the EBIS used to generate the level-set (useful for
                                     // ensuring that neighbor grids are considered when filling LS)

    bool contains_ebs = false;

    void mfix_add_drag_explicit(Real dt);
    void mfix_add_drag_implicit(Real dt);

    void mfix_calc_drag_particle(Real time);

    void mfix_add_gravity_and_gp(Real dt);

    void mfix_compute_MAC_velocity_at_faces(Real time,
                                            Vector< std::unique_ptr<MultiFab> >& vel,
                                            Vector< std::unique_ptr<MultiFab> >& u_mac,
                                            Vector< std::unique_ptr<MultiFab> >& v_mac,
                                            Vector< std::unique_ptr<MultiFab> >& w_mac);

    void mfix_set_velocity_bcs (Real time,
                                Vector< std::unique_ptr<MultiFab> >& vel,
                                int extrap_dir_bcs);

    void mfix_set_scalar_bcs (Real time,
                              Vector< std::unique_ptr<MultiFab> >& ro_g,
                              Vector< std::unique_ptr<MultiFab> >& trac,
                              Vector< std::unique_ptr<MultiFab> >& ep_g,
                              Vector< std::unique_ptr<MultiFab> >& mu_g);

    void set_velocity_bcs(Real* time,
                          const int lev,
                          FArrayBox& vel_fab,
                          const Box& domain,
                          const int* extrap_dir_bcs);

    void set_scalar_bcs(Real* time,
                        const int lev,
                        FArrayBox& scal_fab,
                        const int comp,
                        const Box& domain);

    void set_vec_bcs(const int lev,
                     FArrayBox& vec_fab,
                     const Box& domain);

    void set_gradp_bcs(const Box& bx,
                       const int lev,
                       FArrayBox& gp_fab,
                       Box& domain);

    void set_mac_velocity_bcs(Real* time,
                              const Box& bx,
                              MFIter* mfi,
                              const int lev,
                              Vector< std::unique_ptr<MultiFab> >& u,
                              Vector< std::unique_ptr<MultiFab> >& v,
                              Vector< std::unique_ptr<MultiFab> >& w,
                              Box& domain);

    void set_ls_inflow(const int lev,
                       FArrayBox& ls_phi_fab,
                       const Box& domain,
                       const int* levelset_nghost,
                       const int& nref,
                       const Real* dx);

    void mfix_compute_slopes(int lev, Real time, MultiFab& Sborder,
                             Vector<std::unique_ptr<MultiFab>>& xslopes_in,
                             Vector<std::unique_ptr<MultiFab>>& yslopes_in,
                             Vector<std::unique_ptr<MultiFab>>& zslopes_in,
                             int slopes_comp);

    void set_bc0(const Box& sbx,
                 MFIter* mfi,
                 const int lev,
                 const Box& domain);

    void set_p0(const Box& bx,
                MFIter* mfi,
                const int lev,
                const Box& domain,
                const Real xlen,
                const Real ylen,
                const Real zlen,
                const int delp_dir_in);

    void mfix_set_bc_mod(const int* pID, const int* pType,
                         const amrex::Real* pLo, const amrex::Real* pHi,
                         amrex::Real* pLoc,
                         amrex::Real* pPg,
                         amrex::Real* pVel);

    void mfix_set_bc_mod_add_mi(const int* pPlane,
                                amrex::Real* xLo, amrex::Real* yLo, amrex::Real* zLo,
                                amrex::Real* xHi, amrex::Real* yHi, amrex::Real* zHi,
                                amrex::Real* pPg, amrex::Real* pVel);

    static DragType m_drag_type;

    static amrex::Real tcoll_ratio; // input parameter defaulted to 50 used in file src_des/init_collision.f90

    template <typename F>
    void mfix_calc_particle_beta(F DragFunc, Real time);

    void mfix_compute_ugradu(Box& bx,
                             Vector< std::unique_ptr<MultiFab> >& conv,
                             const int conv_comp,
                             Vector< std::unique_ptr<MultiFab> >& state,
                             const int state_comp, const int n_comp,
                             Vector< std::unique_ptr<MultiFab> >& xslopes,
                             Vector< std::unique_ptr<MultiFab> >& yslopes,
                             Vector< std::unique_ptr<MultiFab> >& zslopes,
                             const int slopes_comp,
                             Vector< std::unique_ptr<MultiFab> >& u_mac,
                             Vector< std::unique_ptr<MultiFab> >& v_mac,
                             Vector< std::unique_ptr<MultiFab> >& w_mac,
                             MFIter* mfi,
                             Box& domain,
                             const int lev,
                             const bool is_conservative);

    void mfix_compute_ugradu_eb(Box& bx,
                                Vector< std::unique_ptr<MultiFab> >& conv,
                                const int conv_comp,
                                Vector< std::unique_ptr<MultiFab> >& state,
                                const int state_comp, const int n_comp,
                                Vector< std::unique_ptr<MultiFab> >& xslopes,
                                Vector< std::unique_ptr<MultiFab> >& yslopes,
                                Vector< std::unique_ptr<MultiFab> >& zslopes,
                                const int slopes_comp,
                                Vector< std::unique_ptr<MultiFab> >& u_mac,
                                Vector< std::unique_ptr<MultiFab> >& v_mac,
                                Vector< std::unique_ptr<MultiFab> >& w_mac,
                                MFIter* mfi,
                                Array< const MultiCutFab*,AMREX_SPACEDIM>& areafrac,
                                Array< const MultiCutFab*,AMREX_SPACEDIM>& facecent,
                                const amrex::MultiFab* volfrac,
                                const amrex::MultiCutFab* bndrycent,
                                Box& domain,
                                const EBCellFlagFab& flags,
                                const int lev,
                                const bool is_conservative);

     Real volWgtSum (int lev, const MultiFab & mf, int comp, bool local=false);

     /*

    Vector<MultiFab> * GetVecOfMultiFabs (Vector<std::unique_ptr<MultiFab>> a)
    {
        Vector<MultiFab> r;
        r.reserve(a.size());
        for (const auto& x : a)
        {
            MultiFab b(x.get());
            r.push_back(b);
        }
        return &r;
    }

    Vector<MultiFab>* get_vel_g() {return GetVecOfMultiFabs(vel_g); }
    Vector<MultiFab>* get_ep_g() {return GetVecOfMultiFabs(ep_g); }
    Vector<MultiFab>* get_p_g() {return GetVecOfMultiFabs(p_g); }
    Vector<MultiFab>* get_ro_g() {return GetVecOfMultiFabs(ro_g); }
    Vector<MultiFab>* get_mu_g() {return GetVecOfMultiFabs(mu_g); }
    Vector<MultiFab>* get_diveu() {return GetVecOfMultiFabs(diveu); }
    Vector<MultiFab>* get_gradp_g() {return GetVecOfMultiFabs(gp); }
    Vector<MultiFab>* get_vort() {return GetVecOfMultiFabs(vort); }
      */
    MFIXParticleContainer::ParticleContainer* thePC () { return pc.get(); }

protected:

     //! Tagging cells for refinement
     virtual void ErrorEst(int lev, TagBoxArray & tags, Real time, int ngrow) override;

     //! Make a new level using provided BoxArray and DistributionMapping and
     //! fill with interpolated coarse level data. Called by AmrCore::regrid.
     virtual void MakeNewLevelFromCoarse(int lev, Real time, const BoxArray & ba,
                                         const DistributionMapping & dm) override
     {
         amrex::Abort("MakeNewLevelFromCoarse: To be implemented");
     }

     //! Remake an existing level using provided BoxArray and
     //! DistributionMapping and fill with existing fine and coarse data. Called
     //! by AmrCore::regrid.
     virtual void RemakeLevel(int lev, Real time, const BoxArray & ba,
                              const DistributionMapping & dm) override
     {
         amrex::Abort("RemakeLevel: To be implemented");
     }

     //! Delete level data.  Called by AmrCore::regrid.
     virtual void ClearLevel(int lev) override
     {
         amrex::Abort("ClearLevel: To be implemented");
     }

     void mfix_init_fluid(int is_restarting, Real dt, Real stop_time);
     void mfix_set_bc0();
     void mfix_set_p0();

     static Real gp0[3];
     static Real gravity[3];

     BcList bc_list;

     amrex::Cuda::ManagedVector<amrex::Real> m_bc_u_g;
     amrex::Cuda::ManagedVector<amrex::Real> m_bc_v_g;
     amrex::Cuda::ManagedVector<amrex::Real> m_bc_w_g;
     amrex::Cuda::ManagedVector<amrex::Real> m_bc_t_g;
     amrex::Cuda::ManagedVector<amrex::Real> m_bc_ep_g;
     amrex::Cuda::ManagedVector<amrex::Real> m_bc_p_g;

     //! The particles see an inflow face as a solid wall; thus instead of
     //! creating a separate EBfactory for particles, we simply modify the
     //! level-set near inflow to see it as a wall
     void mfix_set_ls_near_inflow();

     void mfix_project_velocity   ();
     void mfix_initial_iterations ( Real dt, Real stop_time );
     void mfix_apply_projection   ( Real dt, Real scaling_factor, bool proj_2 );

     void apply_MAC_projection ( Vector< std::unique_ptr<MultiFab> >& u,
                                 Vector< std::unique_ptr<MultiFab> >& v,
                                 Vector< std::unique_ptr<MultiFab> >& w,
                                 Vector< std::unique_ptr<MultiFab> >& ep,
                                 const Vector< std::unique_ptr<MultiFab> >& ro,
                                 Real time, int steady_state);

     void set_MC_velocity_bcs ( int lev,
                                Vector< std::unique_ptr<MultiFab> >& u,
                                Vector< std::unique_ptr<MultiFab> >& v,
                                Vector< std::unique_ptr<MultiFab> >& w,
                                Real time);

     void FillPatchVel    (int lev, Real time, MultiFab & mf, int icomp, int ncomp,
                           const Vector<BCRec> & bcr);
     void GetDataVel      (int lev, Real time, Vector<MultiFab *> & data,
                           Vector<Real> & datatime);

     void FillPatchScalar(int lev, Real time, MultiFab & mf, int icomp, 
                          const Vector<BCRec> & bcr);
     void GetDataScalar (int lev, Real time, Vector<MultiFab *> & data, int icomp, 
                         Vector<Real> & datatime);

     void mfix_calc_volume_fraction(Real & sum_vol);
     void mfix_calc_particle_beta(Real time);
     void mfix_calc_drag_fluid(Real time);

     void AllocateArrays (int lev);

     void RegridArrays (int lev);

     void RegridLevelSetArray (int a_lev);

     void avgDown (int crse_lev, const MultiFab & S_fine, MultiFab & S_crse);

private:

     void make_eb_general();
     void make_eb_box();
     void make_eb_cylinder();
     void make_eb_hopper();
     void make_eb_cyclone();
     void make_eb_air_reactor();
     void make_eb_proto_clr();
     void make_eb_hourglass();
     void make_eb_clr();
     void make_eb_clr_riser();
     void make_eb_regular();

     void MakeBCArrays();

     void check_data();

     void MakeNewLevelFromScratch(int lev, Real time, const BoxArray & new_grids,
                                  const DistributionMapping & new_dmap) override;

     void ReMakeNewLevelFromScratch(int lev, const BoxArray & new_grids,
                                    const DistributionMapping & new_dmap);

     void WriteCheckHeader(const std::string & name, int nstep, Real dt, Real time) const;

     void WriteJobInfo(const std::string & dir ) const;

     static void GotoNextLine(std::istream & is);

     void EvolveFluid(int nstep, Real & dt, Real & time, Real stop_time, Real drag_timing);

     // Projection-related methods
     void mfix_solve_poisson_equation( Vector< std::unique_ptr<MultiFab> >& phi,
                                       Vector< std::unique_ptr<MultiFab> >& rhs,
                                       Vector< std::unique_ptr<MultiFab> >& b,
                                       Vector< std::unique_ptr<MultiFab> >& fluxes);

     // Solve implicit diffusion
     void solve_diffusion_equation(Vector< Vector< std::unique_ptr<MultiFab> > >& b,
                                   Vector< std::unique_ptr<MultiFab> >& phi,
                                   Vector< std::unique_ptr<MultiFab> >& rhs,
                                   int bc_lo[], int bc_hi[],
                                   Real dt);

     void check_for_nans(int lev);

     void mfix_compute_ugradu_predictor(Vector< std::unique_ptr<MultiFab> >& conv_u,
                                        Vector< std::unique_ptr<MultiFab> >& conv_s,
                                        Vector< std::unique_ptr<MultiFab> >& vel,
                                        Vector< std::unique_ptr<MultiFab> >& ro_g,
                                        Vector< std::unique_ptr<MultiFab> >& trac,
                                        Real time);

     void mfix_compute_ugradu_corrector(Vector< std::unique_ptr<MultiFab> >& conv_u,
                                        Vector< std::unique_ptr<MultiFab> >& conv_s,
                                        Vector< std::unique_ptr<MultiFab> >& vel,
                                        Vector< std::unique_ptr<MultiFab> >& ro_g,
                                        Vector< std::unique_ptr<MultiFab> >& trac,
                                        Real time);

     void mfix_compute_divtau(Vector< std::unique_ptr<MultiFab> >& divtau,
                              Vector< std::unique_ptr<MultiFab> >& vel);

     int steady_state_reached(Real dt, int iter);

     void mfix_apply_predictor(Vector< std::unique_ptr<MultiFab> >& conv_u_old,
                               Vector< std::unique_ptr<MultiFab> >& conv_s_old,
                               Vector< std::unique_ptr<MultiFab> >& divtau_old,
                               Real time, Real dt, bool proj_2);
     void mfix_apply_corrector(Vector< std::unique_ptr<MultiFab> >& conv_u_old,
                               Vector< std::unique_ptr<MultiFab> >& conv_s_old,
                               Vector< std::unique_ptr<MultiFab> >& divtau_old,
                               Real time, Real dt, bool proj_2);

     void mfix_diffuse_velocity       (Real time, Real dt);
     void mfix_diffuse_velocity_tensor(Real time, Real dt);

     void mfix_compute_diveu(Real time);

     void mfix_print_max_vel(int lev);
     void mfix_print_max_gp (int lev);

     // Make a level 0 grids covering the whole domain.  It does NOT install the new grids.
     BoxArray MakeBaseGrids() const;
     void ChopGrids(const Box& domain, BoxArray& ba, int target_size) const;

     void mfix_set_bc_type(int lev);

     Real mfix_norm1 ( const Vector< std::unique_ptr<MultiFab>>& mf, int lev, int comp );
     Real mfix_norm1 ( MultiFab& mf, int lev, int comp );
     Real mfix_max   ( MultiFab& mf, int lev, int comp );
     Real mfix_min   ( MultiFab& mf, int lev, int comp );
     Real mfix_norm0 ( const Vector< std::unique_ptr<MultiFab>>& mf, int lev, int comp );
     Real mfix_norm0 ( MultiFab& mf, int lev, int comp );
     Real mfix_norm0 ( const Vector< std::unique_ptr<MultiFab>>& mf1,
                       const Vector< std::unique_ptr<MultiFab>>& mf2,
                       int lev, int comp1, int comp2 );
     Real mfix_norm0 ( MultiFab& mf1, MultiFab& mf2, int lev, int comp1, int comp2 );

     static int nlev;

     // Particle container
     std::unique_ptr<MFIXParticleContainer> pc;

     // Unit vectors in Cartesian space
     static IntVect e_x;
     static IntVect e_y;
     static IntVect e_z;

     // Debug counter used for tracking number of level-set MultiFabs used in
     // intersection/union
     int ct_ls_mf;

     // Boundary conditions types
     Vector< std::unique_ptr<IArrayBox> > bc_ilo;
     Vector< std::unique_ptr<IArrayBox> > bc_ihi;
     Vector< std::unique_ptr<IArrayBox> > bc_jlo;
     Vector< std::unique_ptr<IArrayBox> > bc_jhi;
     Vector< std::unique_ptr<IArrayBox> > bc_klo;
     Vector< std::unique_ptr<IArrayBox> > bc_khi;

     // Boundary conditions flags
     Vector< std::unique_ptr<iMultiFab> > flag;

     // Void fraction
     Vector< std::unique_ptr<MultiFab> > ep_g ;
     Vector< std::unique_ptr<MultiFab> > ep_go;

     // Gas pressure fraction
     Vector< std::unique_ptr<MultiFab> > p_g ;
     Vector< std::unique_ptr<MultiFab> > p_go;

     // Gas density
     Vector< std::unique_ptr<MultiFab> > ro_g ;
     Vector< std::unique_ptr<MultiFab> > ro_go;

     // Tracer in gas
     Vector< std::unique_ptr<MultiFab> > trac;
     Vector< std::unique_ptr<MultiFab> > trac_o;

     // Gas velocity
     Vector< std::unique_ptr<MultiFab> > vel_g ;
     Vector< std::unique_ptr<MultiFab> > vel_go;

     // Base state pressure
     Vector< std::unique_ptr<MultiFab> > p0_g;

     // Pressure gradients
     Vector< std::unique_ptr<MultiFab> > gp ;

     // Molecular viscosity
     Vector< std::unique_ptr<MultiFab> > mu_g ;

     // Cell-based
     Vector< std::unique_ptr<MultiFab> > vort;

     Vector< std::unique_ptr<MultiFab> > drag;

     // Level-Set Data => used for particle-wall collisions and fluid
     // reconstruction in particle drag calculation. NOTE: this has at least 2
     // levels: 0) fluid reconstruction, 1..N) particle-wall (refined) collisions
     Vector< std::unique_ptr<MultiFab> > level_sets;

     // These are multi-component multifabs
     Vector< std::unique_ptr<MultiFab> > xslopes_u;
     Vector< std::unique_ptr<MultiFab> > yslopes_u;
     Vector< std::unique_ptr<MultiFab> > zslopes_u;
     Vector< std::unique_ptr<MultiFab> > xslopes_s;
     Vector< std::unique_ptr<MultiFab> > yslopes_s;
     Vector< std::unique_ptr<MultiFab> > zslopes_s;

     // div (ep_g * u)
     Vector< std::unique_ptr<MultiFab> > diveu;

     // RHS for MAC solve
     Vector< std::unique_ptr<MultiFab> > mac_rhs;

     // Solution for MAC projection
     Vector< std::unique_ptr<MultiFab> > mac_phi;

     // RHS for diffusive tensor solve
     Vector< std::unique_ptr<MultiFab> > diff_rhs;

     // Solution for diffusion solves
     Vector< std::unique_ptr<MultiFab> > diff_phi;

     //
     Vector< std::unique_ptr<MultiFab> > fp;

     // Cell-centered coefficients b in nodal solve
     Vector< std::unique_ptr<MultiFab> > bcoeff_nd;

     // Pressure increment
     Vector< std::unique_ptr<MultiFab> > phi_nd;

     // used if load_balance_type == "KnapSack"
     Vector<std::unique_ptr<MultiFab> > particle_cost;
     Vector<std::unique_ptr<MultiFab> > fluid_cost;

     int steady_state = 0;

     int solve_fluid;
     int solve_dem;

     bool advect_density = false;
     bool advect_tracer  = false;

     // Options to control mfix verbosity level
     int m_verbose = 0;

     // ***************************************************************
     // Nodal solve
     // ***************************************************************

     // Verbosity
     int  nodal_mg_verbose = 0;
     int  nodal_mg_cg_verbose = 0;

     // Control MLMG behavior
     int  nodal_mg_maxiter = 100;
     int  nodal_mg_cg_maxiter = 100;
     Real nodal_mg_rtol = 1.0e-11;
     Real nodal_mg_atol = 1.0e-14;
     std::string nodal_bottom_solver_type;

     // Max coarsening level
     int  nodal_mg_max_coarsening_level = 100;

     // Linear operator
     std::unique_ptr<MLNodeLaplacian> nodal_matrix;

     // Solver
     std::unique_ptr<MLMG> nodal_solver;

     // ***************************************************************
     // MAC solve
     // ***************************************************************

     // Verbosity
     int  mac_mg_verbose = 0;
     int  mac_mg_cg_verbose = 0;

     // Control MLMG behavior
     int  mac_mg_maxiter = 200;
     int  mac_mg_cg_maxiter = 200;
     Real mac_mg_rtol = 1.0e-11;
     Real mac_mg_atol = 1.0e-14;
     std::string mac_bottom_solver_type;

     // Max coarsening level
     int  mac_mg_max_coarsening_level = 100;
     // ***************************************************************

     // ***************************************************************
     // Diffusion solve
     // ***************************************************************

     // Verbosity
     int  diff_mg_verbose = 0;
     int  diff_mg_cg_verbose = 0;

     // Control MLMG behavior
     int  diff_mg_maxiter = 100;
     int  diff_mg_cg_maxiter = 100;
     Real diff_mg_rtol = 1.0e-11;
     Real diff_mg_atol = 1.0e-14;
     std::string diff_bottom_solver_type;

     // Max coarsening level
     int  diff_mg_max_coarsening_level = 100;
     // ***************************************************************

     // Tolerance to check for steady state --
     //    this must be specified in the inputs file
     Real steady_state_tol;

     // Maximum number of iterations to steady state --
     //    this default may be over-written in the inputs file
     int  steady_state_maxiter = 100000000;

     int call_udf;
     bool dual_grid = false;

     // If true then print the name of the routine we are in
     bool ooo_debug = false;

     // Parameters to control load balancing
     static std::string particle_init_type;
     static std::string load_balance_type;
     static std::string knapsack_weight_type;
     static int load_balance_fluid;
     static int knapsack_nmax;

     // Options to control time stepping
     Real cfl = 0.5;
     Real fixed_dt;
     Real dt_min  = 0.0;           // Min dt allowed
     Real dt_max  = 1.e14;         // Max dt allowed

     int particle_max_grid_size_x = -1;
     int particle_max_grid_size_y = -1;
     int particle_max_grid_size_z = -1;

     int removeOutOfRange = 1; // Flag to remove out-of-range particles

     static EBSupport m_eb_support_level;

     Periodicity p0_periodicity;

     Real sum_vol_orig;

     Real covered_val = 1.e40;

     //
     // Options to control the computing of particle eulerian velocities
     //

     Vector<int> avg_p_g  ;
     Vector<int> avg_ep_g ;
     Vector<int> avg_vel_g;

     Vector<int> avg_vel_p;

     Vector<Real> avg_region_x_e;
     Vector<Real> avg_region_x_w;
     Vector<Real> avg_region_y_s;
     Vector<Real> avg_region_y_n;
     Vector<Real> avg_region_z_b;
     Vector<Real> avg_region_z_t;

     // Flags for saving fluid data in plot files
     int plt_vel_g   = 1;
     int plt_ep_g    = 1;
     int plt_p_g     = 0;
     int plt_ro_g    = 0;
     int plt_mu_g    = 0;
     int plt_diveu   = 0;
     int plt_volfrac = 0;
     int plt_gradp_g = 0;
     int plt_vort    = 0;

     // Total number of variables to write in plot file
     int pltVarCount = 0;

     // Flags for saving particle data. By default, we have all flags on,
     // we turn off what we don't want in the init IO routine. This is somewhat
     // different from what we do with the fluid.
     Vector<int> write_real_comp = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
     Vector<int> write_int_comp = {1, 1};

     //
     // Here we set the number of ghost nodes of the field arrays.
     // Since this is  a class attribute, it will be propagated to
     // all the member functions.
     //
     // DO NOT USE  A LOCAL VARIABLE WITH THE SAME NAME
     //

     // Four ghost cells are required when using EB
     const int nghost = 4;

     const int m_eb_basic_grow_cells = nghost;
     const int m_eb_volume_grow_cells = nghost;
     const int m_eb_full_grow_cells = nghost;

     bool do_initial_proj    = true;
     int  initial_iterations = 3;

     /***************************************************************************
      *                                                                         *
      * I/O stuff                                                               *
      *                                                                         *
      **************************************************************************/

     // Variables to simplify checkpoint IO
     Vector< const Vector< std::unique_ptr<MultiFab> >* > vectorVars;
     Vector< std::string > vecVarsName;

     Vector< const Vector< std::unique_ptr<MultiFab> >* > chkscalarVars;
     Vector< std::string > chkscaVarsName;

     /***************************************************************************
      *                                                                         *
      * Used for fillpatching ...                                               *
      *                                                                         *
      **************************************************************************/

     Vector<Real> t_old;
     Vector<Real> t_new;

     Vector<BCRec> bcs_ls; // used by the level-set fill-patch

     Vector<BCRec> bcs_u; // For velocity components
     Vector<BCRec> bcs_s; // For scalars
     Vector<BCRec> bcs_f; // For generic first-order extrapolation


     /***************************************************************************
      *                                                                         *
      * EB Data (including level-set data)                                      *
      *                                                                         *
      **************************************************************************/

     //! EB levels representing fluid boundary conditions
     Vector<const EB2::Level *> eb_levels;
     //! EB levels representing particle boundary conditions (same as
     //! `mfix::eb_levels` but might include additional walls at MI BCs).
     Vector<const EB2::Level *> particle_eb_levels;

     //! EB factory that lives on the fluid grids
     Vector< std::unique_ptr<amrex::EBFArrayBoxFactory> > ebfactory;
     //! EB factory that lives on the particle grids
     Vector< std::unique_ptr<amrex::EBFArrayBoxFactory> > particle_ebfactory;

     // Max level at which to solve the fluid equations
     int amr_max_level = 0;

     bool mfix_update_ebfactory (int a_lev);

};

inline std::string mfix::get_load_balance_type()
{
    return load_balance_type;
}

#endif
